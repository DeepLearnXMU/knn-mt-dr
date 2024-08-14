import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import f1_score, recall_score, accuracy_score, precision_score
from torch.nn.modules.loss import _Loss
from fairseq.criterions.label_smoothed_cross_entropy import (
    label_smoothed_nll_loss,
)

class WeightedLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, probability, target, weight, reduction='sum'):
        target_prob = probability.gather(-1, target.unsqueeze(1))
        alpha = weight[target]
        focal_loss = - alpha * target_prob.log()

        if reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss.mean()


class FasterCriterion(_Loss):
    def __init__(
        self,
        with_adaptive,
        padding_idx,
    ):
        super().__init__()

        self.with_adaptive = with_adaptive
        self.padding_idx = padding_idx
        self.criterion = WeightedLoss()

    
    def forward(self, model, sample, skip_extr=None, validation=False):
        combined_lprobs, nmt_lprobs, knn_lprobs, knn_lambda, knn_alpha \
            = model.train_forward(**sample["net_input"])
        target = sample["target"].view(-1)
        knn_lambda = knn_lambda.view(-1)
        knn_alpha = knn_alpha.view(-1, knn_alpha.size(-1))
        combined_lprobs = combined_lprobs.view(-1, combined_lprobs.size(-1))
        nmt_lprobs = nmt_lprobs.view(-1, combined_lprobs.size(-1))
        index = F.gumbel_softmax(knn_alpha, tau=0.1, hard=True)

        combined_loss, _ = label_smoothed_nll_loss(
            combined_lprobs,
            target,
            0,
            ignore_index=self.padding_idx,
            reduce=False,
        )
        nmt_loss, _ = label_smoothed_nll_loss(
            nmt_lprobs,
            target,
            0,
            ignore_index=self.padding_idx,
            reduce=False,
        )
        combiner_loss = torch.stack((combined_loss.squeeze(dim=-1), nmt_loss.squeeze(dim=-1)), dim=-1)
        translation_loss = torch.sum(index * combiner_loss, dim=-1)

        # nmt_target_position = torch.where(
            # nmt_lprobs.sort(dim=-1, descending=True).indices == target.unsqueeze(1))[1]
        _, nmt_max_token = torch.max(nmt_lprobs, dim=-1)
        _, combined_max_token = torch.max(combined_lprobs, dim=-1)
        nmt_target_position = torch.where(
            nmt_lprobs.sort(dim=-1, descending=True).indices == target.unsqueeze(1))[1]
        alpha_target = torch.ones_like(target).long()
        alpha_target.masked_fill_(nmt_target_position != 0, 1)
        alpha_target.masked_fill_(nmt_max_token != combined_max_token, 0)


        pos_cls_num = alpha_target.sum().item()
        neg_cls_num = len(alpha_target) - pos_cls_num
        rate = pos_cls_num / (neg_cls_num+pos_cls_num)
        weight = torch.tensor([rate, 1 - rate],
                              device='cuda').float()
        alpha_loss = self.criterion(
            knn_alpha, alpha_target, weight=weight.unsqueeze(1), reduction='sum')

        alpha_target = alpha_target.cpu()
        knn_alpha_index = torch.argmax(knn_alpha, dim=1).cpu()

        accuracy = accuracy_score(y_true=alpha_target, y_pred=knn_alpha_index)
        precision = precision_score(y_true=alpha_target, y_pred=knn_alpha_index, average='macro')
        recall = recall_score(y_true=alpha_target, y_pred=knn_alpha_index, average='macro')

        if self.with_adaptive:
            loss = translation_loss.sum() + alpha_loss + combiner_loss.sum()
        else:
            loss = translation_loss.sum() + alpha_loss
            
        
        logging_output = {
            "loss": loss.data,
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
                "knn_alpha": knn_alpha[..., 1].data,
                "knn_lambda": knn_lambda.data,
            }
        
        return loss, logging_output, skip_extr
