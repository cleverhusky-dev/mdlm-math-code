"""One-step diffu-GRPO surrogate for the project's time-independent MDLM.

The generated answer is a target, never visible context for probability
estimation. Every completion slot (including EOS/PAD) is masked. The prompt
is independently masked using a reproducible private generator, and the same
masked view must be supplied to current, old and reference policies.

These are one-step surrogate probabilities, not exact likelihoods of the
multi-step sampler. As in d1, estimation defaults to untempered logits even
when rollout decoding uses a different temperature. There is no causal shift.
"""
from __future__ import annotations

import math
from numbers import Real

import torch
from torch import nn

from .diffusion import validate_temperature


def _scalar(value, name, *, minimum, maximum=None):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real scalar")
    value = float(value)
    if not math.isfinite(value) or value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f"Invalid {name}: {value}")
    return value


def _target_mask(mask):
    if not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool or mask.ndim != 2:
        raise ValueError("target_mask must be a boolean [batch,length] tensor")
    if not mask.numel() or not bool(mask.any(dim=1).all()):
        raise ValueError("Each answer must contain at least one target token")


def make_prompt_mask(target_mask: torch.Tensor, p: float, seed: int) -> torch.Tensor:
    """Select prompt tokens only, without advancing global CPU/CUDA RNG state.

    All prefix positions are eligible, including BOS and separator tokens.
    Generate this mask once for the full local rollout, then slice it for
    microbatches. Recreating it separately for each slice changes the view.
    """
    _target_mask(target_mask)
    p = _scalar(p, "prompt masking probability", minimum=0.0, maximum=1.0)
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**63:
        raise ValueError("mask seed must be an integer in [0, 2**63)")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    draw = torch.rand(target_mask.shape, generator=generator, device="cpu")
    return (draw.to(target_mask.device) < p) & ~target_mask


class PolicyScorer(nn.Module):
    """Expose differentiable gathered log probabilities through DDP.forward.

    The runner owns eval mode: dropout must be disabled in all three policies
    while gradients remain enabled for the current policy. Do not call this
    through ``ddp.module`` during gradient-bearing updates.
    """
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, input_ids, target_mask, prompt_mask, *, temperature=1.0):
        _target_mask(target_mask)
        temperature = validate_temperature(temperature)
        if (not isinstance(input_ids, torch.Tensor) or input_ids.dtype != torch.long
                or input_ids.shape != target_mask.shape or input_ids.device != target_mask.device):
            raise ValueError("input_ids must be a matching int64 [batch,length] tensor")
        if (not isinstance(prompt_mask, torch.Tensor) or prompt_mask.dtype != torch.bool
                or prompt_mask.shape != target_mask.shape or prompt_mask.device != target_mask.device):
            raise ValueError("prompt_mask must be a matching boolean tensor")
        if bool((prompt_mask & target_mask).any()):
            raise ValueError("prompt_mask must not select completion positions")
        if bool(target_mask[:, 0].any()) or bool((target_mask[:, :-1] & ~target_mask[:, 1:]).any()):
            raise ValueError("target_mask must be a completion suffix after a nonempty prefix")
        if (bool((input_ids < 0).any()) or bool((input_ids >= self.model.vocab_size).any())
                or bool((input_ids == self.model.mask_index).any())):
            raise ValueError("Generated input must contain valid non-MASK token IDs")
        masked = input_ids.masked_fill(target_mask | prompt_mask, self.model.mask_index)
        log_probs = self.model.log_probs(masked, temperature=temperature)
        if log_probs.shape != (*input_ids.shape, self.model.vocab_size):
            raise ValueError("Model log probabilities have an unexpected shape")
        gathered = log_probs.gather(-1, input_ids.unsqueeze(-1)).squeeze(-1)
        result = torch.where(target_mask, gathered, torch.zeros_like(gathered))
        if not bool(torch.isfinite(result).all()):
            raise FloatingPointError("Non-finite one-step target log probability")
        return result


def group_advantages(rewards: torch.Tensor, group_size: int) -> torch.Tensor:
    """Center rewards within each contiguous question group; no std scaling."""
    if isinstance(group_size, bool) or not isinstance(group_size, int) or group_size < 2:
        raise ValueError("group_size must be an integer >= 2")
    if (not isinstance(rewards, torch.Tensor) or rewards.ndim != 1 or not rewards.numel()
            or rewards.numel() % group_size or not rewards.is_floating_point()):
        raise ValueError("rewards must be a floating vector of complete question groups")
    if not bool(torch.isfinite(rewards).all()):
        raise FloatingPointError("Non-finite rollout reward")
    groups = rewards.detach().reshape(-1, group_size)
    return (groups - groups.mean(dim=1, keepdim=True)).reshape(-1)


def policy_loss(logps, old_logps, ref_logps, advantages, target_mask,
                clip_epsilon: float, kl_beta: float):
    """Return an unnormalized loss and sums for exact global token averaging.

    ``loss_sum`` retains autograd. Every other returned tensor is detached.
    The runner divides by the global target-token count and compensates for
    DDP's gradient averaging; do not average per-microbatch means equally.
    Old/reference log probabilities and sequence advantages are constants.
    All-equal reward groups have zero policy advantage but can retain KL loss.
    """
    _target_mask(target_mask)
    clip_epsilon = _scalar(clip_epsilon, "clip_epsilon", minimum=0.0, maximum=1.0)
    if clip_epsilon == 0 or clip_epsilon == 1:
        raise ValueError("clip_epsilon must lie strictly between 0 and 1")
    kl_beta = _scalar(kl_beta, "kl_beta", minimum=0.0)
    for name, value in (("logps", logps), ("old_logps", old_logps), ("ref_logps", ref_logps)):
        if (not isinstance(value, torch.Tensor) or not value.is_floating_point()
                or value.shape != target_mask.shape or value.device != target_mask.device):
            raise ValueError(f"{name} must be a matching floating [batch,length] tensor")
        if not bool(torch.isfinite(value[target_mask]).all()):
            raise FloatingPointError(f"Non-finite target {name}")
    if (not isinstance(advantages, torch.Tensor) or not advantages.is_floating_point()
            or advantages.shape != (target_mask.shape[0],) or advantages.device != target_mask.device):
        raise ValueError("advantages must be one floating value per generated answer")
    if not bool(torch.isfinite(advantages).all()):
        raise FloatingPointError("Non-finite advantage")

    # Ignore prompt entries before exponentiation (0 * NaN would still be NaN).
    current = torch.where(target_mask, logps.float(), 0.0)
    old = torch.where(target_mask, old_logps.detach().float(), 0.0)
    reference = torch.where(target_mask, ref_logps.detach().float(), 0.0)
    advantage = advantages.detach().float().unsqueeze(1)
    ratio = torch.exp(current - old)
    clipped_ratio = ratio.clamp(1.0 - clip_epsilon, 1.0 + clip_epsilon)
    unclipped = ratio * advantage
    clipped = clipped_ratio * advantage
    per_token_policy = -torch.minimum(unclipped, clipped)
    delta = reference - current
    # expm1 is more accurate than exp(delta) - 1 when policies nearly agree.
    per_token_kl = torch.expm1(delta) - delta
    per_token_loss = per_token_policy + kl_beta * per_token_kl
    for name, value in (("probability ratio", ratio), ("sampled KL", per_token_kl),
                        ("policy loss", per_token_loss)):
        if not bool(torch.isfinite(value[target_mask]).all()):
            raise FloatingPointError(f"Non-finite {name}; no silent numerical clipping is applied")
    loss_sum = per_token_loss[target_mask].sum(dtype=torch.float32)
    if not bool(torch.isfinite(loss_sum)):
        raise FloatingPointError("Non-finite summed policy loss")
    with torch.no_grad():
        active_clip = (unclipped > clipped) & target_mask
        outside = ((ratio < 1.0 - clip_epsilon) | (ratio > 1.0 + clip_epsilon)) & target_mask
        return {
            "loss_sum": loss_sum,
            "target_tokens": target_mask.sum(dtype=torch.int64),
            "policy_sum": per_token_policy[target_mask].sum(dtype=torch.float32),
            "kl_sum": per_token_kl[target_mask].sum(dtype=torch.float32),
            "clip_count": active_clip.sum(dtype=torch.int64),
            "outside_clip_count": outside.sum(dtype=torch.int64),
            "ratio_sum": ratio[target_mask].sum(dtype=torch.float32),
        }
