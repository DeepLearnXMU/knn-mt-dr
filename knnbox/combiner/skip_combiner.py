import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from knnbox.combiner.utils import calculate_knn_prob, calculate_combined_prob

class SkipCombiner(nn.Module):
    def __init__(self, probability_dim, token_feature_path, max_k, knn_lambda=0.7, temperature=10, mode=None,
                 output_projection_path=None, with_adaptive=False, **kwargs):
        super().__init__()

        self.mode = mode
        self.device = 'cuda'
        self.kwargs = kwargs
        self.probability_dim = probability_dim
        self.token_feature_path = token_feature_path
        self.max_k = max_k
        self.knn_lambda = knn_lambda
        self.knn_temperature = temperature
        self.with_adaptive = with_adaptive
        if self.mode == 'train':
            output_projection_ckp = torch.load(output_projection_path)
            self.output_projection = nn.Linear(1024, probability_dim, bias=False)
            self.output_projection.load_state_dict(output_projection_ckp)


        self.meta_network = MetaNetwork(token_feature_path=token_feature_path, max_k=max_k, with_adaptive=with_adaptive)
        self.mask_for_distance = self._generate_mask_for_distance(self.max_k, self.device)
        self.cache_knn_lambda = None
        self.cache_spare_knn_prob = None
        self.cache_knn_tgt = None
        self.relu = nn.ReLU()
    
    def forward(
        self,
        nmt_prob,
        knn_tgt,
        knn_dist,
        knn_alpha = None,
        retrieve_index = None,
    ):  
        
        sparse_B, K = knn_tgt.size()
        B = nmt_prob.size(0)
        if self.with_adaptive:
            part_knn_lambda, k_probs = self.get_knn_inf(
                knn_dist=knn_dist,
                knn_alpha=knn_alpha[..., 1],
                knn_tgt=knn_tgt,
            )
            R_K = k_probs.size(-1)
            distances = knn_dist.unsqueeze(-2).expand(sparse_B, R_K, K)
            distances = distances * self.mask_for_distance  # [B, R_K, K]
            temperature = self.knn_temperature
            distances = - distances / temperature
            knn_weight = torch.softmax(distances, dim=-1)  # [B, R_K, K]
            spare_knn_prob = torch.matmul(k_probs.unsqueeze(-2), knn_weight).squeeze(-2)  # [B, K]
        else:
            part_knn_lambda = self.knn_lambda
            knn_dist = - knn_dist / self.knn_temperature
            spare_knn_prob = torch.softmax(knn_dist, dim=-1)
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
        return combined_prob


    def get_knn_inf(self, knn_alpha, knn_dist, knn_tgt):
        return self.meta_network.get_knn_inf(knn_alpha, knn_dist, knn_tgt)


    def train_forward(
        self,
        query,
        knn_dist,
        knn_tgt,
        attn,
    ):
        nmt_prob = torch.softmax(self.output_projection(query), dim=-1)
        norm = torch.norm(query, dim=1)
        knn_alpha = self.meta_network(nmt_prob, attn, is_train=True, norm=norm)
        if self.with_adaptive:
            knn_lambda, k_probs = self.get_knn_inf(
                knn_dist=knn_dist,
                knn_alpha=knn_alpha[..., 1],
                knn_tgt=knn_tgt,
            )
            B, K = knn_tgt.size()
            R_K = k_probs.size(-1)

            distances = knn_dist.unsqueeze(-2).expand(B, R_K, K)
            distances = distances * self.mask_for_distance  # [B, R_K, K]

            temperature = self.knn_temperature
            distances = - distances / temperature

            knn_weights = torch.softmax(distances, dim=-1)  # [B, R_K, K]
            knn_weights = torch.matmul(k_probs.unsqueeze(-2), knn_weights).squeeze(-2)  # [B, K]
        else:
            scaled_dists = - knn_dist / self.knn_temperature
            knn_weights = torch.softmax(scaled_dists, dim=-1)
            B, K = knn_tgt.size()
            knn_lambda = torch.empty([B, 1], device=self.device, dtype=torch.float32)
            knn_lambda.fill_(self.knn_lambda)

        knn_prob = torch.zeros(B, self.probability_dim, device=self.device)
        knn_prob.scatter_add_(dim=-1, index=knn_tgt, src=knn_weights)

        combined_prob = knn_lambda * knn_prob + (1 - knn_lambda) * nmt_prob

        nmt_prob = nmt_prob.log()
        knn_prob = knn_prob.log()
        combined_prob = combined_prob.log()
        return combined_prob, nmt_prob, knn_prob, knn_lambda, knn_alpha
        

    @staticmethod
    def _generate_mask_for_distance(max_k, device):
        k_mask = torch.empty((max_k, max_k)).fill_(999.)
        k_mask = torch.triu(k_mask, diagonal=1) + 1
        power_index = torch.tensor([pow(2, i) - 1 for i in range(0, int(math.log(max_k, 2)) + 1)])
        k_mask = k_mask[power_index]
        k_mask.requires_grad = False
        k_mask = k_mask.to(device)
        return k_mask

    def dump(self, path):
        config = {}
        config["probability_dim"] = self.probability_dim
        config["token_feature_path"] = self.token_feature_path
        config['max_k'] = self.max_k
        config["knn_lambda"] = self.knn_lambda
        config["temperature"] = self.knn_temperature
        config["with_adaptive"] = self.with_adaptive
        for k, v in self.kwargs.items():
            config[k] = v
        state_dict = {
            "config": config,
            "model": self.meta_network.state_dict()
        }
        torch.save(state_dict, path)


    @classmethod
    def load(cls, path):
        print(f'load combiner from {path}')
        state_dict = torch.load(path)
        config = state_dict["config"]
        model = cls(**config)
        model.meta_network.load_state_dict(state_dict["model"])
        return model

    
class MetaNetwork(nn.Module):
    def __init__(self, token_feature_path, max_k, with_adaptive, midsize=32, dropout_rate=0.2):
        super().__init__()
        self.device = 'cuda'
        self.token_feature_path = token_feature_path
        self.max_k = max_k
        self.with_adaptive = with_adaptive
        assert self.token_feature_path != None

        self.bn = nn.BatchNorm1d(3)
        self.alpha_func = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
            nn.Softmax(dim=-1),
        )
        nn.init.normal_(self.alpha_func[0].weight, mean=0, std=0.1)
        nn.init.normal_(self.alpha_func[-2].weight)


        if self.with_adaptive:
            self.midsize = midsize
            self.dropout_rate = dropout_rate
            self.distance_to_k = nn.Sequential(
                nn.Linear(self.max_k * 2, self.midsize),
                nn.Tanh(),
                nn.Dropout(p=self.dropout_rate),
                nn.Linear(self.midsize, int(math.log(self.max_k, 2)) + 1),
                nn.Softmax(dim=-1),
            )
            nn.init.normal_(self.distance_to_k[0].weight[:, :self.max_k], mean=0, std=0.01)
            nn.init.normal_(self.distance_to_k[0].weight[:, self.max_k:], mean=0, std=0.1)
            nn.init.normal_(self.distance_to_k[-2].weight)

            self.distance_to_lambda = nn.Sequential(
                nn.Linear(self.max_k * 2, self.midsize),
                nn.Tanh(),
                nn.Dropout(p=self.dropout_rate),
                nn.Linear(self.midsize, 1),
                nn.Sigmoid(),
            )
            nn.init.xavier_normal_(self.distance_to_lambda[0].weight[:, :self.max_k], gain=0.01)
            nn.init.xavier_normal_(self.distance_to_lambda[0].weight[:, self.max_k:], gain=0.1)
            nn.init.xavier_normal_(self.distance_to_lambda[-2].weight)

            mask_for_label_count = torch.empty((self.max_k, self.max_k)).fill_(1)
            mask_for_label_count = torch.triu(mask_for_label_count, diagonal=1).bool()
            mask_for_label_count.requires_grad = False
            self.mask_for_label_count = mask_for_label_count.to(self.device)

    def forward(self, nmt_prob, attn, norm, is_train=False):
        nmt_max_prob, nmt_max_token = torch.max(nmt_prob, dim=-1, keepdim=True)
        nmt_max_token.squeeze_(-1)
        alpha_knn_input = torch.cat([nmt_max_prob, attn, norm.unsqueeze(dim=-1)],
                                    dim=-1)
        alpha_knn_input = self.bn(alpha_knn_input)
        knn_alpha = self.alpha_func(alpha_knn_input)
        if is_train:
            return knn_alpha
        return knn_alpha[..., 1]
    

    def get_knn_inf(self, knn_alpha, knn_dist, knn_tgt):
        label_counts = self._get_label_count_segment(knn_tgt)
        network_inputs = torch.cat((knn_dist.detach(), label_counts.detach().float()), dim=-1)
        k_probs = self.distance_to_k(network_inputs)
        k_lambda = self.distance_to_lambda(network_inputs).clamp_max(0.99)

        return k_lambda, k_probs

    def _get_label_count_segment(self, vals):
        r""" this function return the label counts for different range of k nearest neighbor
            [[0:0], [0:1], [0:2], ..., ]
        """
        B, K = vals.size()
        expand_vals = vals.unsqueeze(dim=-2).expand(B, K, K)
        expand_vals = expand_vals.masked_fill(self.mask_for_label_count, value=-1)
        labels_sorted, _ = expand_vals.sort(dim=-1)
        labels_sorted[:, :, 1:] *= ((labels_sorted[:, :, 1:] - labels_sorted[:, :, :-1]) != 0).long()
        retrieve_label_counts = labels_sorted.ne(0).sum(-1)
        retrieve_label_counts[:, :-1] -= 1
        return retrieve_label_counts
