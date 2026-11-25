"""One synchronized two-GPU policy, with a separate CPU control channel.

Training collectives use NCCL on CUDA. Validation, checkpoints and execution
receipts use Gloo, so peers waiting for rank-zero work do not leave a pending
NCCL operation. CPU execution exists for explicit one/two-process fixtures.
All ranks must call collective methods in the same order.
"""
from __future__ import annotations

from datetime import timedelta
import math
from numbers import Real
import os
import traceback

import torch
import torch.distributed as dist

from .runtime import capture_rng


class DistributedContext:
    """Own the process groups for one single-node training invocation.

    Use ``torchrun --standalone --nproc_per_node=2`` for production CUDA jobs.
    ``device_name='cpu'`` also permits a single-process fixture without a
    rendezvous. ``gather_objects`` preserves rank order and does not flatten.
    """

    def __init__(self, device_name="cuda", *, timeout_seconds=86400):
        if device_name not in {"cuda", "cpu"}:
            raise ValueError("device_name must be 'cuda' or explicit fixture 'cpu'")
        if (isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, Real)
                or not math.isfinite(timeout_seconds) or timeout_seconds <= 0):
            raise ValueError("Distributed timeout must be positive finite seconds")
        self.rank = int(os.environ.get("RANK", "0"))
        self.world = int(os.environ.get("WORLD_SIZE", "1"))
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if self.world not in (1, 2) or not 0 <= self.rank < self.world:
            raise ValueError("RL supports one-process CPU fixtures or single-node two-process DDP")
        if not 0 <= self.local_rank < self.world:
            raise ValueError("LOCAL_RANK is outside the single-node process range")
        if int(os.environ.get("LOCAL_WORLD_SIZE", str(self.world))) != self.world:
            raise ValueError("Multi-node RL is not supported")
        if device_name == "cuda" and self.world != 2:
            raise ValueError("CUDA RL requires two ranks; launch with torchrun --nproc_per_node=2")
        if dist.is_initialized():
            raise RuntimeError("DistributedContext must own a fresh process group")

        self.device = (torch.device("cuda", self.local_rank)
                       if device_name == "cuda" else torch.device("cpu"))
        self.control = None
        self._owns_default = False
        self._closed = False
        if self.device.type == "cuda":
            if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
                raise RuntimeError("Two CUDA GPUs are required; use CPU only for explicit fixtures")
            torch.cuda.set_device(self.device)

        if self.world > 1:
            if not dist.is_available() or not dist.is_gloo_available():
                raise RuntimeError("PyTorch Gloo support is required for the control group")
            backend = "nccl" if self.device.type == "cuda" else "gloo"
            if backend == "nccl" and not dist.is_nccl_available():
                raise RuntimeError("This PyTorch build lacks CUDA NCCL support")
            timeout = timedelta(seconds=float(timeout_seconds))
            try:
                dist.init_process_group(backend, init_method="env://", rank=self.rank,
                                        world_size=self.world, timeout=timeout)
                self._owns_default = True
                self.control = dist.new_group(ranks=list(range(self.world)),
                                              backend="gloo", timeout=timeout)
            except Exception:
                self.close()
                raise

    def _check_open(self):
        if self._closed:
            raise RuntimeError("DistributedContext is closed")
        if self.world > 1 and (not dist.is_initialized() or self.control is None):
            raise RuntimeError("Distributed process groups are no longer initialized")

    def rank_zero(self, fn):
        """Run a callback only on rank zero and propagate its result or error.

        Results should be small CPU-serializable metadata. Python exceptions
        are broadcast; a killed rank is bounded by the process-group timeout.
        """
        self._check_open()
        message = [None]
        if self.rank == 0:
            try:
                message[0] = {"result": fn(), "error": None}
            except Exception as error:
                message[0] = {
                    "result": None,
                    "error": {"type": type(error).__name__, "message": str(error),
                              "traceback": traceback.format_exc()},
                }
        if self.world > 1:
            dist.broadcast_object_list(message, src=0, group=self.control)
        record = message[0]
        if record["error"] is not None:
            error = record["error"]
            if error["type"] == "TimeoutError":
                raise TimeoutError(error["message"])
            raise RuntimeError(
                f"Rank-zero stage failed: {error['type']}: {error['message']}\n"
                f"{error['traceback']}"
            )
        return record["result"]

    def gather_objects(self, value):
        """Return a list of per-rank objects on every rank, preserving groups."""
        self._check_open()
        if self.world == 1:
            return [value]
        result = [None] * self.world
        dist.all_gather_object(result, value, group=self.control)
        return result

    def gather_rng(self):
        return self.gather_objects(capture_rng())

    def any_true(self, value):
        """Synchronize control flags on CPU, independently of training NCCL."""
        self._check_open()
        flag = torch.tensor(int(bool(value)), dtype=torch.int64, device="cpu")
        if self.world > 1:
            dist.all_reduce(flag, op=dist.ReduceOp.MAX, group=self.control)
        return bool(flag.item())

    def close(self):
        """Release owned groups once, without adding a shutdown barrier."""
        if self._closed:
            return
        self._closed = True
        if self._owns_default and dist.is_initialized():
            try:
                if self.control is not None:
                    dist.destroy_process_group(self.control)
            finally:
                dist.destroy_process_group()
        self.control = None
        self._owns_default = False

    def __enter__(self):
        self._check_open()
        return self

    def __exit__(self, *_):
        self.close()
