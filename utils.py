import logging
import math
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_fscore_support
import torch
import torch.nn as nn
import torch.nn.functional as F
from Model.LoRALinear import LoRALinear


def inject_lora_to_encoder(model, r, alpha):
    for name, module in model.encoder.named_children():
        # print(f"[LoRA] Replacing {name} Linear: in={module.in_features}, out={module.out_features}")
        if isinstance(module, nn.Linear):
            new_module = LoRALinear(
                module.in_features,
                module.out_features,
                r=r,
                alpha=alpha,
                pretrained_weight=module.weight
            )
            setattr(model.encoder, name, new_module)
        elif isinstance(module, nn.Sequential) or isinstance(module, nn.ModuleList):
            inject_lora_to_encoder(module, r, alpha)
        else:
            continue
    return model

def aucPerformance(score, labels):
    roc_auc = roc_auc_score(labels, score)
    ap = average_precision_score(labels, score)
    return roc_auc, ap

def F1Performance(score, target):
    normal_ratio = (target == 0).sum() / len(target)
    score = np.squeeze(score)
    threshold = np.percentile(score, 100 * normal_ratio)
    pred = np.zeros(len(score))
    pred[score > threshold] = 1

    precision, recall, f1, _ = precision_recall_fscore_support(target, pred, average='binary')
    return f1

def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "%(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])
    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger
