import torch
import torch.nn.functional as F

# @torch.compile
def _get_bin_range(logits: torch.Tensor, 
                  ks: torch.Tensor, 
                  normalized_deltas: torch.Tensor, 
                  epsilon: float = 1e-5) -> torch.Tensor:
    # k denote the number of elements in the first bin
    logits_sorted, _ = logits.sort(dim=-1, descending=True)
    deltas = logits_sorted[:, 0] - torch.gather(logits_sorted, dim=-1, index=ks[:, None]).squeeze(-1)
    logits_range = logits_sorted[:, 0] - logits_sorted[:, -1]
    M = (logits_range / deltas).max().ceil().long()
    bin_logits = torch.full((logits.shape[0], M), -float('inf'), device=logits.device)
    bin_range = torch.full((logits.shape[0], M), -float('inf'), device=logits.device)
    bin_mask = torch.zeros((logits.shape[0], M), device=logits.device)

    for i in range(logits.shape[0]):
        bin_size = (logits_range[i] // deltas[i]).ceil().long()
        # bin_range[i, :bin_size] = torch.linspace(logits_sorted[i, -1] - epsilon, logits_sorted[i, 0], bin_size)
        bin_range[i, :bin_size] = -torch.linspace(- logits_sorted[i, 0], - logits_sorted[i, -1] + epsilon, bin_size)
        bin_mask[i, :bin_size - 1] = 1
        bin_logits[i, :bin_size - 1] = (-torch.arange(bin_size - 1, device=logits.device)) * normalized_deltas[i] + logits_sorted[i, 0]
        # bin_logits[i, :bin_size - 1] = (-torch.arange(bin_size - 1, device=logits.device)).flip(dims=[0]) * deltas[i] + logits_sorted[i, 0]
    intra_bin_probs = F.softmax(bin_logits, dim=-1)
    del bin_logits
    return bin_range, bin_mask, intra_bin_probs

# @torch.compile
def _get_bin_logprobs_torch(logits, 
                            ks: torch.Tensor, 
                            normalized_deltas: torch.Tensor, 
                            epsilon: float = 1e-5) -> torch.Tensor:
    """
    bin_range is always the left boundary of the bin.
    """
    bin_range, bin_mask, intra_bin_probs = _get_bin_range(logits, ks, normalized_deltas, epsilon)
    
    N, V = logits.shape
    # 1. Assign each logit to a bin
    # bin_assignments will have shape [N, V] with values in [0, M-1]
    bin_assignments = torch.empty_like(logits, dtype=torch.long)
    
    _, M = bin_range.shape
    
    # right = False means bin_assignments[j] = i if  bin_range[i - 1] < logits[j] <= bin_range[i]
    bin_assignments = torch.searchsorted(- bin_range, - logits, right=False).long() - 1

    mask_m = bin_assignments[:, None, :] == torch.arange(M, device=logits.device)[None, :, None] # [N, M, V]

    logits_per_bin = torch.where(mask_m, logits.unsqueeze(1).repeat(1, M, 1), -float('inf')) # [N, M, V]
    logits_per_bin = logits_per_bin - logits_per_bin.max(dim=-1, keepdim=True)[0]
    
    bin_probs = F.softmax(logits_per_bin, dim=-1) # shape [N, M, V]
    del logits_per_bin
    
    bin_sample_id = torch.full((N, M), -1, device=logits.device)
    
    for i in range(N):
        for j in range(M):
            if bin_mask[i, j] == 1:
                if torch.isnan(bin_probs[i, j, :]).sum() > 0:
                    bin_sample_id[i, j] = -1
                else:
                    bin_sample_id[i, j] = torch.multinomial(bin_probs[i, j, :], num_samples=1)

    return bin_sample_id, intra_bin_probs
