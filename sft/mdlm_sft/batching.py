"""Deterministic no-duplication batches, with a resumable optimizer-step cursor."""
from __future__ import annotations

import math
import torch
from torch.utils.data import DataLoader, Sampler


class GlobalUpdateSampler(Sampler):
    """Each DataLoader batch is one global update, before rank/micro partitioning.

    A private generator prevents constructing an iterator or resuming it from
    consuming the model RNG. num_workers=0 intentionally avoids prefetch state.
    """
    def __init__(self, size, batch_size, seed, epoch, start_update=0):
        self.size, self.batch_size = size, batch_size
        self.seed, self.epoch, self.start_update = seed, epoch, start_update
        if size < 1 or batch_size < 1 or not 0 <= start_update <= math.ceil(size / batch_size):
            raise ValueError("Invalid data cursor")

    def __len__(self):
        return math.ceil(self.size / self.batch_size) - self.start_update

    def __iter__(self):
        g = torch.Generator().manual_seed(self.seed + self.epoch)
        order = torch.randperm(self.size, generator=g).tolist()
        for start in range(self.start_update * self.batch_size, self.size, self.batch_size):
            yield order[start:start + self.batch_size]


def update_loader(dataset, global_batch_size, seed, epoch, start_update=0):
    return DataLoader(dataset, batch_sampler=GlobalUpdateSampler(
        len(dataset), global_batch_size, seed, epoch, start_update), num_workers=0,
        generator=torch.Generator().manual_seed(seed + epoch), pin_memory=False)


def rank_slice(count, rank, world_size):
    # Balanced contiguous rank-major partition; no DistributedSampler duplication.
    base, extra = divmod(count, world_size)
    start = rank * base + min(rank, extra)
    return start, start + base + int(rank < extra)


def step_counts(dataset, global_batch_size, epochs):
    per_epoch = len(update_loader(dataset, global_batch_size, 0, 0))
    return per_epoch, per_epoch * epochs, math.ceil(per_epoch / 2)


def lr_multiplier(completed_steps, total_steps, warmup_steps):
    if total_steps < 1 or not 0 <= completed_steps <= total_steps:
        raise ValueError("Invalid scheduler progress")
    if completed_steps < warmup_steps:
        return completed_steps / max(1, warmup_steps)
    return max(0.0, (total_steps - completed_steps) / max(1, total_steps - warmup_steps))
