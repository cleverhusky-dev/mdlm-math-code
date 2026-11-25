"""Conditional continuous-time MDLM, adapted from pinned s-flm algo/trainer_base.

No autoregressive shift, no causal attention, no prompt loss. The loss API
returns both a sum and a target-token denominator so the caller can aggregate
validation correctly. PAD and EOS remain ordinary target tokens.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from numbers import Real

import torch
from torch import nn


@dataclass(frozen=True)
class ModelConfig:
    hidden_size: int = 768
    cond_dim: int = 128
    n_blocks: int = 12
    n_heads: int = 12
    dropout: float = 0.1
    length: int = 512
    # Published TinyGSM checkpoint metadata uses 0 (confirmed by user),
    # unlike the later pinned repository's default 1e-3. sampling_eps differs.
    noise_eps: float = 0.0
    sampling_eps: float = 1e-3
    antithetic_sampling: bool = True


class LogLinear(nn.Module):
    """Same parameter-free schedule as upstream noise_schedules.LogLinear."""
    def __init__(self, eps: float = 0.0):
        super().__init__()
        if not 0 <= eps < 1:
            raise ValueError("noise eps must be in [0,1)")
        self.eps = eps

    def alpha_t(self, t):
        return self.eps + (1 - self.eps) * (1 - t)

    def forward(self, t):
        return -(1 - self.eps) * torch.ones_like(t), self.alpha_t(t)


def sample_times(n: int, device, *, global_batch_size: int | None = None,
                 offset: int = 0, sampling_eps: float = 1e-3,
                 antithetic: bool = True):
    """Upstream antithetic strata, optionally select this rank/micro's slice.

    Like upstream, a new full random vector is drawn on each micro forward.
    For a full update offset = rank * accumulation * micro_batch +
    accumulation_index * micro_batch. The caller supplies a compact offset
    and actual group size for an incomplete final update.
    """
    size = n if global_batch_size is None else global_batch_size
    if n < 1 or offset < 0 or offset + n > size:
        raise ValueError("Invalid antithetic batch slice")
    u = torch.rand(size, device=device)
    if antithetic:
        u = (u / size + torch.arange(size, device=device) / size) % 1
    return ((1 - sampling_eps) * u + sampling_eps)[offset:offset + n]


def validate_temperature(temperature: float) -> float:
    """Accept a positive finite real scalar, never a boolean or tensor."""
    if isinstance(temperature, bool) or not isinstance(temperature, Real):
        raise ValueError("temperature must be a positive finite real number")
    try:
        temperature = float(temperature)
    except (OverflowError, ValueError) as error:
        raise ValueError("temperature must be a positive finite real number") from error
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be a positive finite real number")
    return temperature


def subs_log_probs(logits, xt, mask_index: int, *, temperature: float = 1.0):
    """SUBS: forbid MASK output; exact copying of already visible tokens.

    Log-normalization uses FP32 for numerical stability under BF16 autocast.
    T=1 retains the original operations, including the finite -1e6 MASK
    convention, exactly. Other temperatures scale token logits in FP32 before
    normalization. Centering non-MASK logits avoids positive overflow at low
    temperatures; excluding MASK after scaling prevents it from reappearing
    at high temperatures. Visible tokens remain exact copies for every T.
    """
    temperature = validate_temperature(temperature)
    logits = logits.float().clone()
    if temperature == 1.0:
        logits[..., mask_index] += -1_000_000.0
    else:
        logits[..., mask_index] = -torch.inf
        logits -= logits.amax(dim=-1, keepdim=True)
        # A positive Python float can be smaller than a representable FP32
        # denominator. Divide in stages so maxima stay 0, rather than 0/0.
        remaining = temperature
        minimum = torch.finfo(logits.dtype).tiny
        while remaining < minimum:
            logits.div_(minimum)
            remaining /= minimum
        logits.div_(remaining)
        logits[..., mask_index] = -torch.inf
    log_probs = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
    visible = xt != mask_index
    log_probs[visible] = -1_000_000.0
    log_probs[visible, xt[visible]] = 0
    return log_probs


class MDLM(nn.Module):
    def __init__(self, backbone: nn.Module, vocab_size: int, mask_index: int,
                 config: ModelConfig | None = None):
        super().__init__()
        self.config = config or ModelConfig()
        self.backbone = backbone
        self.noise = LogLinear(self.config.noise_eps)
        self.vocab_size = vocab_size
        self.mask_index = mask_index
        self.num_tokens = self.config.length
        self.time_conditioning = False
        self.diffusion_type = "absorbing"
        self.loss_type = "elbo"

    @property
    def device(self):
        return next(self.backbone.parameters()).device

    @property
    def dtype(self):
        return next(self.backbone.parameters()).dtype

    def log_probs(self, xt, *, temperature: float = 1.0):
        temperature = validate_temperature(temperature)
        sigma = torch.zeros(xt.shape[0], device=xt.device, dtype=torch.float32)
        logits = self.backbone(None, xt, sigma, None)
        return subs_log_probs(logits, xt, self.mask_index, temperature=temperature)

    def corrupt(self, input_ids, target_mask, t, corruption_uniform=None):
        if input_ids.ndim != 2 or target_mask.shape != input_ids.shape:
            raise ValueError("Expected matching [batch,length] input and target mask")
        if t.shape != (input_ids.shape[0],) or not ((t > 0) & (t <= 1)).all():
            raise ValueError("Expected one diffusion time in (0,1] per sample")
        if ((input_ids < 0) | (input_ids >= self.vocab_size)).any():
            raise ValueError("Input token outside checkpoint vocabulary")
        if (input_ids == self.mask_index).any():
            raise ValueError("Clean inputs must not contain MASK")
        if corruption_uniform is None:
            corruption_uniform = torch.rand(input_ids.shape, device=input_ids.device)
        if corruption_uniform.shape != input_ids.shape:
            raise ValueError("Corruption uniforms must match input shape")
        _, alpha = self.noise(t)
        masked = (corruption_uniform < 1 - alpha[:, None]) & target_mask.bool()
        return torch.where(masked, self.mask_index, input_ids), masked

    def forward(self, input_ids, target_mask, *, t=None, corruption_uniform=None,
                global_batch_size=None, offset=0):
        if t is None:
            t = sample_times(input_ids.shape[0], input_ids.device,
                             global_batch_size=global_batch_size, offset=offset,
                             sampling_eps=self.config.sampling_eps,
                             antithetic=self.config.antithetic_sampling)
        xt, masked = self.corrupt(input_ids, target_mask, t, corruption_uniform)
        log_probs = self.log_probs(xt)
        nll = -log_probs.gather(-1, input_ids.unsqueeze(-1)).squeeze(-1)
        # alpha'/(1-alpha)=-1/t for the fixed log-linear schedule. SUBS
        # makes all unmasked nll exactly zero; explicit mask documents scope.
        token_loss = nll * masked / t[:, None]
        numerator = token_loss.sum(dtype=torch.float32)
        denominator = target_mask.sum(dtype=torch.int64)
        return {"loss_sum": numerator, "target_tokens": denominator,
                "loss": numerator / denominator.clamp_min(1),
                "masked_tokens": masked.sum(), "t": t.detach()}


def build_model(tokenizer, config: ModelConfig | dict | None = None):
    from .transformer import DIT

    config = ModelConfig(**config) if isinstance(config, dict) else (config or ModelConfig())
    # Mirrors upstream VocabSizeTokenizerWrapper for SmolLM: PAD is added
    # before selecting a new MASK ID. tokenizer.vocab_size alone is wrong.
    if (tokenizer.vocab_size, len(tokenizer), tokenizer.bos_token_id,
            tokenizer.eos_token_id, tokenizer.pad_token_id) != (49152, 49153, 0, 0, 49152):
        raise ValueError("Tokenizer does not match the audited SmolLM vocabulary")
    if tokenizer.mask_token_id is not None:
        raise ValueError("Audited tokenizer has no native MASK token")
    vocab_size, mask_index = len(tokenizer) + 1, len(tokenizer)
    architecture = {
        "model": {**asdict(config), "type": "dit", "softcap": 0.0},
        "algo": {"causal_attention": False, "adaLN": True},
    }
    return MDLM(DIT(architecture, vocab_size), vocab_size, mask_index, config)
