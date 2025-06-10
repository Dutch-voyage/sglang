import torch
import torch.nn.functional as F

def _entropy(probs: torch.Tensor):
    return -torch.sum(probs * torch.log(probs), dim=-1)


def _varentropy(probs: torch.Tensor, entropy: torch.Tensor = None):
    if entropy is None:
        entropy = _entropy(probs)
    return torch.sum(probs * (torch.log(probs)) ** 2, dim=-1) - entropy ** 2