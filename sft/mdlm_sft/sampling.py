"""Fixed ancestral sampling, traced to pinned s-flm samplers.py.

The final transition is one of `steps`, not an extra model evaluation.
Sampling remains stochastic at that step. Only the target region is sampled.
"""
from __future__ import annotations

import torch
import time

from .diffusion import validate_temperature


def sample_categorical(probs):
    # Exact upstream Gumbel/exponential-race formulation and RNG shape.
    noise = 1e-10 - (torch.rand_like(probs) + 1e-10).log()
    return (probs / noise).argmax(dim=-1)


def absorbing_transition(probs, xt, alpha_s, alpha_t, mask_index, *, final=False):
    sampled = sample_categorical(probs)
    if final:
        should_denoise = torch.ones_like(sampled, dtype=torch.bool)
    else:
        probability = (alpha_s - alpha_t) / (1 - alpha_t)
        should_denoise = torch.rand_like(sampled, dtype=torch.float64) < probability
    return torch.where((xt == mask_index) & should_denoise, sampled, xt)


@torch.no_grad()
def ancestral_sample(model, prefix_ids, *, steps=32, length=512,
                     temperature=1.0, use_float64=True, eps=1e-5, deadline=None,
                     cancel_requested=None):
    """Generate from question prefixes only; never accepts reference code/gold.

    Caller owns eval mode, RNG isolation, autocast, fixed batch/order, and
    deciding which process may sample. Returned tensor includes the prefix.
    `eps=1e-5` matches AncestralSampler.init_state's generation default;
    training's independent sampling_eps is 1e-3.
    """
    temperature = validate_temperature(temperature)
    if steps < 1 or length != model.num_tokens:
        raise ValueError("Invalid fixed ancestral settings")
    if not prefix_ids or any(not 0 < len(p) < length for p in prefix_ids):
        raise ValueError("Each prefix must leave a nonempty generation region")
    xt = torch.full((len(prefix_ids), length), model.mask_index,
                    dtype=torch.long, device=model.device)
    for i, prefix in enumerate(prefix_ids):
        prefix = torch.as_tensor(prefix, dtype=torch.long, device=model.device)
        if ((prefix < 0) | (prefix >= model.vocab_size) | (prefix == model.mask_index)).any():
            raise ValueError("Invalid token in generation prefix")
        xt[i, :len(prefix)] = prefix
    start = min(map(len, prefix_ids))
    # Keep exactly the upstream linspace and model-parameter dtype for alpha.
    times = torch.linspace(1, eps, steps + 1, device=model.device)
    ones = torch.ones(len(prefix_ids), 1, dtype=model.dtype, device=model.device)
    for index in range(steps):
        if cancel_requested is not None and cancel_requested():
            raise TimeoutError("Cancellation requested between denoising steps")
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError("Generation wall-clock boundary reached between denoising steps")
        final = index == steps - 1
        _, alpha_t = model.noise(times[index] * ones)
        _, alpha_s = model.noise(times[index + 1] * ones)
        if final:
            alpha_s = torch.ones_like(alpha_t)
        log_probs = model.log_probs(xt, temperature=temperature)
        if use_float64:
            log_probs = log_probs.to(torch.float64)
        xt[:, start:] = absorbing_transition(
            log_probs[:, start:].exp(), xt[:, start:], alpha_s, alpha_t,
            model.mask_index, final=final)
        # Visible-token copying already preserves prefixes. Project explicitly
        # as upstream does, including the longer prefixes within this window.
        for i, prefix in enumerate(prefix_ids):
            xt[i, :len(prefix)] = torch.as_tensor(prefix, device=model.device)
    if (xt == model.mask_index).any():
        raise RuntimeError("Ancestral final step left MASK tokens")
    return xt
