import torch
from torch import nn as nn
from torch.nn import functional as F


class KDLoss(nn.Module):

    def __init__(self, temp: float, reduction: str):
        super(KDLoss, self).__init__()

        self.temp = temp
        self.reduction = reduction
        self.kl_loss = nn.KLDivLoss(reduction=reduction)

    def forward(self, teacher_logits: torch.Tensor, student_logits: torch.Tensor):

        student_softmax = F.log_softmax(student_logits / self.temp, dim=-1)
        teacher_softmax = F.softmax(teacher_logits / self.temp, dim=-1)

        kl = nn.KLDivLoss(reduction='none')(student_softmax, teacher_softmax)
        kl = kl.sum() if self.reduction == 'sum' else kl.sum(1).mean()
        kl = kl * (self.temp ** 2)

        return kl

    def __call__(self, *args, **kwargs):
        return super(KDLoss, self).__call__(*args, **kwargs)


class LogitsMatching(nn.Module):

    def __init__(self, reduction: str):
        super(LogitsMatching, self).__init__()
        self.mse_loss = nn.MSELoss(reduction=reduction)

    def forward(self, teacher_logits: torch.Tensor, student_logits: torch.Tensor):
        return self.mse_loss(student_logits, teacher_logits)

    def __call__(self, *args, **kwargs):
        return super(LogitsMatching, self).__call__(*args, **kwargs)