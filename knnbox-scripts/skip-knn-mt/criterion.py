import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import f1_score, recall_score, accuracy_score, precision_score
from torch.nn.modules.loss import _Loss

from fairseq.criterions.label_smoothed_cross_entropy import (
    label_smoothed_nll_loss,
)

class FocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super().__init__()
        self.gamma = gamma

    def forward(self, probability, target, weight, reduction='sum', label_smooth=0.01):
        target_prob = probability.gather(-1, target.unsqueeze(1))
        alpha = weight[target]
        label_smooth_alpha = weight[~target]
        factor = (1 - target_prob) ** self.gamma
        label_smooth_alpha_factor = target_prob ** self.gamma
        focal_loss = - ((1-label_smooth) * alpha * factor * target_prob.log() +
                        label_smooth * label_smooth_alpha * label_smooth_alpha_factor * (1-target_prob).log())

        if reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss.mean()


class MeCriterion(_Loss):
    def __init__(
        self,
        label_smoothing,
        padding_idx,
        alpha_coef=0.,
        norm_coef=2.,
        alpha_mode='v1',
        balance_coef=1.5,
    ):
        super().__init__()

        self.eps = label_smoothing
        self.padding_idx = padding_idx
        self.alpha_coef = alpha_coef
        self.norm_coef = norm_coef
        self.alpha_mode = alpha_mode
        self.balance_coef = balance_coef
        self.focal_criterion = FocalLoss(gamma=2)


    
    def forward(self, model, sample, reduce=True, skip_extr=None, validation=False):
        combined_lprobs, nmt_lprobs, knn_lprobs, knn_lambda, knn_alpha \
            = model.train_forward(**sample["net_input"])
        target = sample["target"].view(-1)
        knn_lambda = knn_lambda.view(-1)
        knn_alpha = knn_alpha.view(-1, knn_alpha.size(-1))
        combined_lprobs = combined_lprobs.view(-1, combined_lprobs.size(-1))
        nmt_lprobs = nmt_lprobs.view(-1, combined_lprobs.size(-1))
        knn_lprobs = knn_lprobs.view(-1, combined_lprobs.size(-1))

        loss, combined_nll_loss = label_smoothed_nll_loss(
            combined_lprobs,
            target,
            self.eps,
            ignore_index=self.padding_idx,
            reduce=False,
        )
        _, nmt_nll_loss = label_smoothed_nll_loss(
            nmt_lprobs,
            target,
            self.eps,
            ignore_index=self.padding_idx,
            reduce=False,
        )
        combined_nll_loss = combined_nll_loss.squeeze()
        nmt_nll_loss = nmt_nll_loss.squeeze()

        nmt_target_position = torch.where(
            nmt_lprobs.sort(dim=-1, descending=True).indices == target.unsqueeze(1))[1]
        knn_target_position = torch.where(
            knn_lprobs.sort(dim=-1, descending=True).indices == target.unsqueeze(1))[1]
            
        knn_target_prob = torch.gather(knn_lprobs, -1, target.unsqueeze(1)).squeeze().exp().detach()
        nmt_target_prob = torch.gather(nmt_lprobs, -1, target.unsqueeze(1)).squeeze().exp().detach()
        
        alpha_target = torch.zeros_like(target).long()

        alpha_target.masked_fill_(nmt_target_position != 0, 1)
        alpha_target.masked_fill_(knn_target_prob < 1e-4, 0)

        pos_cls_num = alpha_target.sum().item()
        neg_cls_num = len(alpha_target) - pos_cls_num

        
        weight = torch.tensor([1.0, (pos_cls_num + neg_cls_num) / pos_cls_num if pos_cls_num != 0 else 1.0],
                                              device='cuda').float()
        
        alpha_loss = self.alpha_coef * self.focal_criterion(
            knn_alpha, alpha_target, weight=weight.unsqueeze(1), reduction='sum')

        alpha_target = alpha_target.cpu()
        knn_alpha_index = torch.argmax(knn_alpha, dim=1).cpu()

        accuracy = accuracy_score(y_true=alpha_target, y_pred=knn_alpha_index)
        precision = precision_score(y_true=alpha_target, y_pred=knn_alpha_index, average='macro')
        recall = recall_score(y_true=alpha_target, y_pred=knn_alpha_index, average='macro')
        

        loss = loss.sum() + alpha_loss

        logging_output = {
            "loss": loss.data,
            "nll_loss": combined_nll_loss.sum().data,
            "alpha_loss": alpha_loss.data if alpha_loss != None else None,
            "lam": knn_lambda.sum().data,
            "alpha": knn_alpha[..., 1].sum().data if knn_alpha != None else None,
            "sample_size": sample["target"].size(0),
            "accuracy": accuracy * sample["target"].size(0),
            "precision": precision * sample["target"].size(0),
            "recall": recall * sample["target"].size(0)
        }

        if validation:
            knn_pos_alpha = knn_alpha[alpha_target == 1, 1].mean() * target.size(0)
            knn_neg_alpha = knn_alpha[alpha_target == 0, 1].mean() * target.size(0)

            logging_output["p_alpha"] = knn_pos_alpha
            logging_output["n_alpha"] = knn_neg_alpha
            

        if skip_extr:
            skip_extr = {
                "combined_nll_loss": combined_nll_loss.data,
                "nmt_nll_loss": nmt_nll_loss.data,
                "knn_alpha": knn_alpha[..., 1].data,
                "knn_lambda": knn_lambda.data,
            }
        
        return loss, logging_output, skip_extr
