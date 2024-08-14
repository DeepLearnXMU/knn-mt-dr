import torch
from torch import nn
import torch.nn.functional as F
import math
import os
from knnbox.common_utils import read_config, write_config
from knnbox.combiner.utils import calculate_combined_prob, calculate_knn_prob

from knnbox.modules import MLPMOE
from collections import OrderedDict


class ELCombiner(nn.Module):
    r""" el-knnlm Combiner """
    def __init__(self, 
                probability_dim,
                hidden_units = 128,
                nlayers = 5,
                dropout = 0.2,
                activation = 'relu',
                temperature = 10,
                **kwargs
                ):
        super().__init__()
        #feature_size = OrderedDict({
        #    'ctxt': 512,
        #    'freq': 1,
        #    'nmt_entropy': 1,
        #    'nmt_max': 1,
        #    'fert': 1,
        #})
        feature_size = OrderedDict({
            'ctxt': 1024,
            'freq': 1,
            'nmt_entropy': 1,
            'nmt_max': 1,
            'fert': 1,
        })

        self.mlpmoe = MLPMOE(
            feature_size=feature_size,
            hidden_units=hidden_units,
            nlayers=nlayers,
            dropout=dropout,
            activation=activation,
        )

        self.probability_dim = probability_dim
        self.temperature = temperature
        self.cache_knn_lambda = None
        self.cache_spare_knn_prob = None
        self.cache_knn_tgt = None
        self.adaptive_combiner = None
        self.mask_for_distance = None

    def load_model(self, path):
        ckp = torch.load(path)
        self.mlpmoe.load_state_dict(ckp)


    def get_knn_prob(self, vals, distances, device="cuda:0"):
        return calculate_knn_prob(vals, distances, self.probability_dim,
                     self.temperature, device)
        
    def get_combined_prob(
        self,
        nmt_prob,
        knn_tgt,
        knn_dist,
        part_knn_lambda,
        retrieve_index = None,
        log_probs=False
    ):
        sparse_B, K = knn_tgt.size()
        B = nmt_prob.size(0)
        if self.adaptive_combiner:
            metak_outputs = self.adaptive_combiner.meta_k_network(vals=knn_tgt.unsqueeze(dim=1), distances=knn_dist.unsqueeze(dim=1))
            k_probs = metak_outputs['k_net_output'].squeeze(dim=1)
            part_knn_lambda = metak_outputs['lambda_net_output'].squeeze(dim=1)
            R_K = k_probs.size(-1)
            distances = knn_dist.unsqueeze(-2).expand(sparse_B, R_K, K)
            if not hasattr(self, "mask_for_distance") or self.mask_for_distance is None:
                self.mask_for_distance = self._generate_mask_for_distance(self.adaptive_combiner.max_k, 'cuda')
            distances = distances * self.mask_for_distance  # [B, R_K, K]
            temperature = self.adaptive_combiner.temperature
            distances = - distances / temperature
            knn_weight = torch.softmax(distances, dim=-1)  # [B, R_K, K]
            spare_knn_prob = torch.matmul(k_probs.unsqueeze(-2), knn_weight).squeeze(-2)  # [B, K]
        else:
            knn_dist = - knn_dist / self.temperature
            spare_knn_prob = torch.softmax(knn_dist, dim=-1)
            part_knn_lambda = part_knn_lambda.unsqueeze(dim=-1)
            spare_knn_prob = part_knn_lambda * spare_knn_prob
        if retrieve_index is not None:
            if self.cache_knn_lambda is None:
                self.cache_knn_lambda = torch.empty([B, 1], device=nmt_prob.device, dtype=torch.float)
                self.cache_spare_knn_prob = torch.empty([B, K], device=nmt_prob.device, dtype=torch.float)
                self.cache_knn_tgt = torch.empty([B, K], device=nmt_prob.device, dtype=torch.long)

            self.cache_knn_lambda.fill_(0.)
            self.cache_spare_knn_prob.fill_(0.)
            self.cache_knn_tgt.fill_(0)

            self.cache_knn_lambda[retrieve_index] = part_knn_lambda
            self.cache_spare_knn_prob[retrieve_index] = spare_knn_prob
            self.cache_knn_tgt[retrieve_index] = knn_tgt

            combined_prob = (1 - self.cache_knn_lambda[:B]) * nmt_prob
        else:
            self.cache_knn_lambda = part_knn_lambda
            self.cache_spare_knn_prob = spare_knn_prob
            self.cache_knn_tgt = knn_tgt

            combined_prob = (1 - self.cache_knn_lambda) * nmt_prob

        combined_prob.scatter_add_(
            dim=-1,
            index=self.cache_knn_tgt[:B],
            src=self.cache_spare_knn_prob[:B],
        )
        if log_probs:
            combined_prob = torch.log(combined_prob)
        return combined_prob


    # def get_combined_prob(self, knn_prob, nmt_prob, knn_lambda, log_probs=False):
    #     combined_probs = knn_prob * knn_lambda + nmt_prob * (1 - knn_lambda)
    #
    #     if log_probs:
    #         combined_probs = torch.log(combined_probs)
    #     return combined_probs


    def get_knn_lambda(self, features):
        B, S, _ = features['ctxt'].size()
        for k in features:
            features[k] = features[k].view(B, S, -1)
        self.knn_lambda = self.mlpmoe(features).exp()[..., 1]
        return self.knn_lambda


    @staticmethod
    def _generate_mask_for_distance(max_k, device):
        k_mask = torch.empty((max_k, max_k)).fill_(999.)
        k_mask = torch.triu(k_mask, diagonal=1) + 1
        power_index = torch.tensor([pow(2, i) - 1 for i in range(0, int(math.log(max_k, 2)) + 1)])
        k_mask = k_mask[power_index]
        k_mask.requires_grad = False
        k_mask = k_mask.to(device)
        return k_mask
