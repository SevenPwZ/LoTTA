import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, r, alpha,dropout=0.0, pretrained_weight=None):
        super().__init__()
        self.r = r
        self.alpha = alpha
        self.scale = alpha / r

        if pretrained_weight is not None:
            self.weight = nn.Parameter(pretrained_weight.clone(), requires_grad=False)
        else:
            self.weight = nn.Parameter(torch.empty(out_features, in_features), requires_grad=False)
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        self.lora_A = nn.Linear(in_features, r, bias=False)
        self.lora_B = nn.Linear(r, out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enable_lora=True):
        base = F.linear(x, self.weight)
        if enable_lora:
            lora = self.lora_B(self.dropout(self.lora_A(x))) * self.scale
            return base + lora
        return base
