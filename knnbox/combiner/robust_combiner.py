import torch
from torch import nn
import torch.nn.functional as F
import math
import os
from knnbox.common_utils import read_config, write_config
from knnbox.combiner.utils import calculate_combined_prob, calculate_knn_prob


class RobustCombiner(nn.Module):
    r""" Robust knn-mt Combiner """
    def __init__(self, 
                max_k,
                probability_dim,
                midsize = 32,
                midsize_dc = 4,
                topk_wp = 8,
                use_diff = False,
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

        # self.skip_print = skip_print
        self.use_diff = use_diff

        self.meta_k_network = MetaKNetwork(
            max_k=self.max_k,
            midsize=self.midsize,
            midsize_dc=self.midsize_dc,
            topk_wp=self.topk_wp,
            use_diff=use_diff,
            **kwargs
        )

    def set_num_updates(self, num_updates):
        """State from trainer to pass along to model at every update."""
        
        def _apply(m):
            if hasattr(m, "set_num_updates") and m != self:
                m.set_num_updates(num_updates)

        self.apply(_apply)

    def get_knn_prob(
        self,
        tgt_index: torch.Tensor,
        knn_dists: torch.Tensor,
        knn_key_feature: torch.Tensor,
        network_probs: torch.Tensor,
        network_select_probs: torch.Tensor,
        device="cuda:0"
    ):
        metak_outputs = self.meta_k_network(
            tgt_index=tgt_index,
            knn_dists=knn_dists,
            knn_key_feature=knn_key_feature,
            network_probs=network_probs,
            network_select_probs=network_select_probs
        )

        self.lambda_ = metak_outputs["knn_lambda"]
        self.tempe = metak_outputs["tempe"]
        self.noise_logit = metak_outputs["noise_logit"]


        knn_prob = torch.zeros(*network_probs.shape, device=device)
        knn_prob.scatter_add_(dim=-1, index=tgt_index, src=metak_outputs["probs"])

        return knn_prob

    def get_combined_prob(self, knn_prob, neural_model_logit, log_probs=False):
        r""" get combined probs of knn_prob and neural_model_prob """

        # if self.skip_print:
        #     k = 8
        #
        #     knn_topk_prob, knn_topk_index = knn_prob.topk(k, dim=-1)
        #     nmt_topk_prob, nmt_topk_index = neural_model_logit.topk(k, dim=-1)
        #
        #     cosine_scores = []
        #     for i in range(knn_topk_index.size(0)):
        #         concat_index = torch.cat([knn_topk_index[i, 0], nmt_topk_index[i, 0]], dim=-1)
        #         unique_index, inverse_indices = torch.unique(concat_index, return_inverse=True, sorted=True)
        #         zeros = torch.zeros_like(concat_index).float()
        #         knn_distribution = zeros.scatter(-1, inverse_indices[:k], knn_topk_prob[i, 0]).unsqueeze(0)
        #         nmt_distribution = zeros.scatter(-1, inverse_indices[-k:], nmt_topk_prob[i, 0]).unsqueeze(0)
        #         score = F.cosine_similarity(knn_distribution, nmt_distribution)
        #         cosine_scores.append(score)
        #         print(f'cosine: {score.item():.4f}')
        #
        #     cosine_scores = torch.cat(cosine_scores, dim=-1)
            # print(f'cosine: {cosine_scores.mean()}')

            # Top1 mask analysis
            # skip_mask = (knn_topk == nmt_topk)
            # print(f'prob: {skip_mask.sum().data} / {knn_topk.numel()}')
            # self.lambda_.masked_fill_(skip_mask, 0.)

        return calculate_combined_prob(knn_prob, neural_model_logit, self.lambda_, log_probs)


    def dump(self, path):
        r""" dump the robust knn-mt to disk """
        # step 1. write config
        config = {}
        config["max_k"] = self.max_k
        config["probability_dim"] = self.probability_dim
        config["midsize"] = self.midsize
        config["midsize_dc"] = self.midsize_dc
        config["use_diff"] = self.use_diff
        for k, v in self.kwargs.items():
            config[k] = v
        write_config(path, config)
        # step 2. save model
        torch.save(self.state_dict(), os.path.join(path, "robust_combiner.pt"))


    @classmethod
    def load(cls, path, use_diff=False):
        r""" load the robust knn-mt from disk """
        config = read_config(path)
        config["use_diff"] = use_diff
        robust_combiner = cls(**config)

        robust_combiner.load_state_dict(torch.load(os.path.join(path, "robust_combiner.pt")))
        return robust_combiner
    

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
        device = "cuda:0",
        use_diff = False,
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
        self.use_diff = use_diff

        if self.use_diff is True:
            self.distance_func = nn.Sequential(
                nn.Linear(self.max_k * 2 + self.topk_wp * 4 - 6, 1),
            )
        else:
            self.distance_func = nn.Sequential(
                nn.Linear(self.max_k * 2 + self.topk_wp, 1), # ? W_6
            )

        self.distance_fc1 = nn.Sequential(
            nn.Linear(2, self.midsize_dc), # ? W_4
            nn.Tanh(),
            nn.Linear(self.midsize_dc, 1), # ? W_3
        )
        
        if self.use_diff is True:
            self.distance_fc2 = nn.Sequential(
                nn.Linear(self.max_k * 5 - 6, self.midsize),
                nn.Tanh(),
                nn.Linear(self.midsize, 2),
            )
        else:
            self.distance_fc2 = nn.Sequential(
                nn.Linear(self.max_k * 2, self.midsize), # ? W_2
                nn.Tanh(),
                nn.Linear(self.midsize, 2), # ? [W_1, W_5]
            )

    def forward(
        self,
        tgt_index: torch.Tensor,
        knn_dists: torch.Tensor,
        knn_key_feature: torch.Tensor,
        network_probs: torch.Tensor,
        network_select_probs: torch.Tensor,
    ):
        B, S, K = knn_dists.size()
        label_counts = self._get_label_count_segment(tgt_index, self.relative_label_count)
        all_key_feature = torch.cat([knn_key_feature.log().unsqueeze(-1), network_select_probs.log().unsqueeze(-1)], -1)
        top_prob, top_idx = torch.topk(network_probs, self.topk_wp)
        noise_logit = self.distance_fc1(all_key_feature).squeeze(-1)

        top_prob_log = top_prob.log()
        if self.use_diff is True:
            sim_lambda = self.distance_func(torch.cat([
                knn_key_feature.log(),
                network_select_probs.log(),
                top_prob_log,
                top_prob_log[...,:-1] - top_prob_log[...,1:],
                top_prob_log[...,:-2] - top_prob_log[...,2:],
                top_prob_log[...,:-3] - top_prob_log[...,3:],
            ], -1))
        else:
            sim_lambda = self.distance_func(torch.cat([
                top_prob_log,
                knn_key_feature.log(),
                network_select_probs.log(),
            ], -1))

        if self.use_diff is True:
            knn_feat = torch.cat([
                label_counts.float(),
                knn_dists,
                knn_dists[...,:-1] - knn_dists[...,1:],
                knn_dists[...,:-2] - knn_dists[...,2:],
                knn_dists[...,:-3] - knn_dists[...,3:],
            ], -1)
        else:
            knn_feat = torch.cat([knn_dists, label_counts.float()], -1)
        lambda_logit = self.distance_fc2(knn_feat.view(B, S, -1))
        knn_lambda = torch.softmax(torch.cat([lambda_logit[:, :, :1], sim_lambda], -1), -1)[:, :, :1]
        tempe = torch.sigmoid(lambda_logit[:, :, 1:2])
        probs = torch.softmax(-knn_dists * tempe + noise_logit, -1)
        
        return {
            "probs": probs,
            "tempe": tempe,
            "noise_logit": noise_logit,
            "knn_lambda": knn_lambda,
        }
    

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


