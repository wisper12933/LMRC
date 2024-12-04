import torch
import torch.nn as nn
import torch.nn.functional as F


class Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_funt = nn.BCEWithLogitsLoss()

    def forward(self, logits, labels):
        # ### BCEloss
        loss = self.loss_funt(logits, labels)
        
        # ### AFloss
        # th_label = torch.zeros_like(labels, dtype=torch.float).to(labels)
        # th_label[:, 0] = 1.0
        # labels[:, 0] = 0.0
  
        # n_mask = 1 - labels
        
        # _, num_class = labels.size()

        # # Rank each class to threshold class TH
        # th_mask = torch.cat( num_class * [logits[:,:1]], dim=1)
        # logit_th = torch.cat([logits.unsqueeze(1), 1.0 * th_mask.unsqueeze(1)], dim=1) 
        # log_probs = F.log_softmax(logit_th, dim=1)
        # probs = torch.exp(F.log_softmax(logit_th, dim=1))

        # # Probability of relation class to be negative (0)
        # prob_0 = probs[:, 1 ,:]
        # prob_0_gamma = torch.pow(prob_0, 1.0)
        # log_prob_1 = log_probs[:, 0 ,:]

        # # Rank TH to negative classes
        # logit2 = logits - (1 - n_mask) * 1e30
        # rank2 = F.log_softmax(logit2, dim=-1)
        # loss1 = - (log_prob_1 * (1 + prob_0_gamma ) * labels) 
        # loss2 = - (rank2 * th_label).sum(1) 
        # loss =  1.0 * loss1.sum(1).mean() + 1.0 * loss2.mean()

        ### ATloss
        # th_label = torch.zeros_like(labels, dtype=torch.float).to(labels)
        # th_label[:, 0] = 1.0
        # labels[:, 0] = 0.0

        # p_mask = labels + th_label
        # n_mask = 1 - labels

        # # Rank positive classes to TH
        # logit1 = logits - (1 - p_mask) * 1e30
        # loss1 = -(F.log_softmax(logit1, dim=-1) * labels).sum(1)

        # # Rank TH to negative classes
        # logit2 = logits - (1 - n_mask) * 1e30
        # loss2 = -(F.log_softmax(logit2, dim=-1) * th_label).sum(1)

        # # Sum two parts
        # loss = loss1 + loss2
        # loss = loss.mean()
        
        return loss

    def get_label(self, logits):
        # Change to binary
        probs = torch.sigmoid(logits)
        output = (probs > 0.5).float()
        return output
