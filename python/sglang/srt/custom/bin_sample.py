import torch
import torch.nn.functional as F
from sglang.srt.custom.entropy import _entropy, _varentropy

def gather_mask(
    bin_lower: torch.Tensor,
    bin_upper: torch.Tensor,
    logits: torch.Tensor,
) -> torch.Tensor:
    return (logits[:, None, :] >= bin_lower[:, :, None]) & (logits[:, None, :] < bin_upper[:, :, None])  # [N, M, V]


# @torch.compile
def get_range(
    logits: torch.Tensor,
    ks: torch.Tensor,
    deltas: torch.Tensor,
):
    # logits [N, V]
    # logits [N, K.max]
    bin_upper = (
        torch.max(logits.ceil(), dim=-1)[0][:, None]
        - torch.arange(torch.max(ks), device=logits.device)[None, :] * deltas[:, None]
    )
    bin_lower = bin_upper - deltas[:, None]
    return bin_lower, bin_upper


def _get_bin_range(
    logits: torch.Tensor,
    probs: torch.Tensor,
    ks: torch.Tensor,
    deltas: torch.Tensor,
) -> torch.Tensor:
    # ks denote the number of top bins considered
    N, V = logits.shape

    bin_lower, bin_upper = get_range(logits, ks, deltas)

    bin_mask = gather_mask(bin_lower, bin_upper, logits)  # [N, M, V]

    M_cols = bin_mask.shape[1]

    # Create column indices: [0, 1, ..., M_cols-1]
    col_indices = torch.arange(M_cols, device=logits.device)

    mask = col_indices[None, :] < ks[:, None]  # Resulting mask shape: (N, M_cols)
    values_for_assignment = (1.0 / ks.to(dtype=logits.dtype))[
        :, None
    ]  # Shape (N, 1) for broadcasting
    bin_weights = torch.where(mask, values_for_assignment, 0)

    bin_probs_sum = (bin_mask * probs[:, None, :]).sum(dim=-1)  # [N, M]

    return (bin_lower, bin_upper), bin_mask, bin_probs_sum, bin_weights

def _get_bin_logprobs_torch(
    logits: torch.Tensor,
    ks: torch.Tensor,
    deltas: torch.Tensor,
    need_eager_token_sampling: torch.Tensor, # [B]
    eager_token_ids: torch.Tensor, # [B]
) -> torch.Tensor:

    probs = F.softmax(logits, dim=-1)
    (bin_lower, bin_upper), bin_mask, bin_probs_sum, bin_weights = _get_bin_range(
        logits,
        probs,
        ks,
        deltas,
    )

    N, V, M = bin_mask.shape

    # probs for sampling, where sum of probs in each bin is 1
    bin_probs = torch.where(bin_mask, probs[:, None, :] / bin_probs_sum[:, :, None], 0)
    # probs for entropy calculation, where sum of probs of all bins is 1
    bin_entropy = _entropy(bin_probs * bin_weights[:, :, None])
    bin_varentropy = _varentropy(bin_probs * bin_weights[:, :, None], bin_entropy)

    placeholder = torch.zeros_like(bin_probs)
    placeholder[:, :, -1] = 1
    bin_probs = torch.where(bin_probs_sum[:, :, None] > 0, bin_probs, placeholder)

    bin_sample_id = torch.vmap(
        torch.multinomial, in_dims=(0, None), randomness="different"
    )(bin_probs, 1).squeeze(-1)
    
    eager_mask = bin_probs[torch.arange(bin_probs.shape[0]), :, eager_token_ids] != 0

    bin_sample_id = torch.where(
        eager_mask,
        eager_token_ids[:, None],
        bin_sample_id,
    )
    
    return bin_sample_id, bin_probs_sum, bin_entropy, bin_varentropy
