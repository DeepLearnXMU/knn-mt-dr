import torch
from torch import nn
import torch.nn.functional as F
import math
import os
from knnbox.common_utils import read_config, write_config

class MeCombiner(nn.Module):
    r""" more efficient knn-mt combiner """
    def __init__(self, 
                max_k,
                probability_dim,
                midsize = 32,
                midsize_dc = 4,
                topk_wp = 8,
                use_entropy = False,
                **kwargs
                ):
        super().__init__()
        
        self.max_k = max_k
        self.probability_dim = probability_dim
        self.midsize = midsize
        self.midsize_dc = midsize_dc
        self.topk_wp = topk_wp
        self.kwargs = kwargs 
        self.mask_for_distance = None
        self.use_entropy = use_entropy

        self.meta_k_network = MetaKNetwork(
            max_k=self.max_k,
            midsize=self.midsize,
            midsize_dc=self.midsize_dc,
            topk_wp=self.topk_wp,
            use_entropy=self.use_entropy,
            **kwargs
        )

    def set_num_updates(self, num_updates):
        """State from trainer to pass along to model at every update."""
        
        def _apply(m):
            if hasattr(m, "set_num_updates") and m != self:
                m.set_num_updates(num_updates)

        self.apply(_apply)
    
    def get_knn_lambda(self, nmt_prob):
        return self.meta_k_network.get_knn_lambda(nmt_prob)

    def get_knn_prob(
        self,
        tgt_index: torch.Tensor,
        knn_dists: torch.Tensor,
        nmt_prob: torch.Tensor,
        device="cuda:0"
    ):
        meta_probs = self.meta_k_network.get_knn_probs(
            tgt_index=tgt_index,
            knn_dists=knn_dists,
        )
        knn_prob = torch.zeros(*nmt_prob.shape, device=device)
        knn_prob.scatter_add_(dim=-1, index=tgt_index, src=meta_probs)
        return knn_prob

    def get_combined_prob(self, knn_prob, nmt_prob, lambda_, log_probs=False):
        r""" get combined probs of knn_prob and neural_model_prob """

        lambda_ = lambda_.view(knn_prob.size(0), knn_prob.size(1), 1)
        combined_prob = knn_prob * lambda_ + nmt_prob * (1 - lambda_)

        if log_probs:
            combined_prob = torch.log(combined_prob)
        return combined_prob


    def dump(self, path):
        r""" dump the robust knn-mt to disk """
        # step 1. write config
        config = {}
        config["max_k"] = self.max_k
        config["probability_dim"] = self.probability_dim
        config["midsize"] = self.midsize
        config["midsize_dc"] = self.midsize_dc
        config["use_entropy"] = self.use_entropy
        for k, v in self.kwargs.items():
            config[k] = v
        write_config(path, config)
        # step 2. save model
        torch.save(self.state_dict(), os.path.join(path, "my_combiner.pt"))


    @classmethod
    def load(cls, path):
        r""" load my knn-mt from disk """
        config = read_config(path)
        my_combiner = cls(**config)

        my_combiner.load_state_dict(torch.load(os.path.join(path, "my_combiner.pt")), strict=False)
        return my_combiner
    

    @staticmethod
    def _generate_mask_for_distance(max_k, device):
        k_mask = torch.empty((max_k, max_k)).fill_(999.)
        k_mask = torch.triu(k_mask, diagonal=1) + 1
        power_index = torch.tensor([pow(2, i) - 1 for i in range(0, int(math.log(max_k, 2)) + 1)])
        k_mask = k_mask[power_index]
        k_mask.requires_grad = False
        k_mask = k_mask.to(device)
        return k_mask


class MetaKNetwork(nn.Module):
    r""" meta k network of robust knn-mt """
    def __init__(
        self,
        max_k = 32,
        midsize = 32,
        midsize_dc = 4,
        topk_wp = 8,
        k_trainable = True,
        lambda_trainable = True,
        temperature_trainable = True,
        relative_label_count = False,
        use_entropy = False,
        device = "cuda:0",
        **kwargs,
    ):
        super().__init__()
        self.max_k = max_k    
        self.k_trainable = k_trainable
        self.lambda_trainable = lambda_trainable
        self.temperature_trainable = temperature_trainable
        self.relative_label_count = relative_label_count
        self.device = device
        self.mask_for_label_count = None
        self.midsize = midsize
        self.midsize_dc = midsize_dc
        self.topk_wp = topk_wp
        self.use_entropy = use_entropy

        # Robust kNN-MT always uses the same configuration
        # ? WP network: S_{NMT}

        distance_inputs_size = self.topk_wp * 4 - 6 + 8
        if self.use_entropy:
            distance_inputs_size += 1

        self.distance_func = nn.Sequential(
            nn.Linear(distance_inputs_size, 1)
        )
        self.distance_emb = nn.Sequential(
            nn.Linear(1024, 1)
        )
        # ? DC network: c_k
        self.distance_fc1 = nn.Sequential(
            nn.Linear(2, self.midsize_dc), # ? W_4
            nn.Tanh(),
            nn.Linear(self.midsize_dc, 1), # ? W_3
        )
        # ? WP network: S_{kNN}  &  WP network: T
        self.distance_fc2 = nn.Sequential(
            nn.Linear(self.max_k * 2, self.midsize), # ? W_2
            nn.Tanh(),
            nn.Linear(self.midsize, 1), # ? [W_1, W_5]
        )
    
    def get_knn_lambda(self, nmt_prob):
        top_prob, top_idx = torch.topk(nmt_prob, self.topk_wp)
        top_prob_log = top_prob.log()

        emb_dist = torch.sigmoid(self.distance_emb(self.embeddings[top_idx])).squeeze(-1)

        distance_inputs = [
            top_prob_log, 
            top_prob_log[...,:-1] - top_prob_log[...,1:],
            top_prob_log[...,:-2] - top_prob_log[...,2:],
            top_prob_log[...,:-3] - top_prob_log[...,3:],
            emb_dist,
        ]
        if self.use_entropy:
            nmt_entropy = torch.sum(- nmt_prob * nmt_prob.log(), dim=-1, keepdim=True)
            distance_inputs.append(nmt_entropy)

        sim_lambda = self.distance_func(torch.cat(distance_inputs, -1))
        knn_lambda = 1 - torch.sigmoid(sim_lambda)
        return knn_lambda

    def get_knn_probs(
        self,
        tgt_index: torch.Tensor,
        knn_dists: torch.Tensor,
    ):
        B, S, K = knn_dists.size()
        label_counts = self._get_label_count_segment(tgt_index, self.relative_label_count)
        knn_feat = torch.cat([knn_dists, label_counts.float()], -1)

        lambda_logit = self.distance_fc2(knn_feat.view(B, S, -1))
        tempe = torch.sigmoid(lambda_logit)

        self.tempe = tempe
        
        probs = torch.softmax(-knn_dists * tempe, -1) 
        return probs
    
    def _get_label_count_segment(self, vals, relative=False):
        r""" this function return the label counts for different range of k nearest neighbor 
            [[0:0], [0:1], [0:2], ..., ]
        """
        # caculate `label_count_mask` only once
        if self.mask_for_label_count is None:
            mask_for_label_count = torch.empty((self.max_k, self.max_k)).fill_(1)
            mask_for_label_count = torch.triu(mask_for_label_count, diagonal=1).bool()
            mask_for_label_count.requires_grad = False
            # [0,1,1]
            # [0,0,1]
            # [0,0,0]
            self.mask_for_label_count = mask_for_label_count.to(vals.device)

        ## TODO: The feature below may be unreasonable
        B, S, K = vals.size()
        expand_vals = vals.unsqueeze(-2).expand(B,S,K,K)
        expand_vals = expand_vals.masked_fill(self.mask_for_label_count, value=-1)
        

        labels_sorted, _ = expand_vals.sort(dim=-1) # [B, S, K, K]
        labels_sorted[:, :, :, 1:] *= ((labels_sorted[:, :, :, 1:] - labels_sorted[:, :, : , :-1]) != 0).long()
        retrieve_label_counts = labels_sorted.ne(0).sum(-1)
        retrieve_label_counts[:, :, :-1] -= 1

        if relative:
            retrieve_label_counts[:, :, 1:] = retrieve_label_counts[:, :, 1:] - retrieve_label_counts[:, :, :-1]
        
        return retrieve_label_counts