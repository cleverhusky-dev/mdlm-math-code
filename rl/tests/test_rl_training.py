"""CPU update fixtures, including real two-process Gloo gradient reduction.

No benchmark or project data is read. Process results live in a system
temporary directory, never in the experiment output directory.
"""
from __future__ import annotations

import copy
from dataclasses import asdict
from datetime import timedelta
from pathlib import Path
import socket
import tempfile
from types import SimpleNamespace
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

from mdlm_rl.diffusion import MDLM, ModelConfig
from mdlm_rl.objective import PolicyScorer, group_advantages, make_prompt_mask, policy_loss
from mdlm_rl.train import cache_logps, token_mask, update_policy


class ContextBackbone(nn.Module):
    """Cheap nonconstant logits that depend on the actually masked prompt."""

    def __init__(self):
        super().__init__()
        self.position = nn.Parameter(torch.linspace(-0.3, 0.4, 88).reshape(8, 11))
        self.context = nn.Parameter(torch.sin(torch.arange(121).float()).reshape(11, 11) * 0.2)
        self.dropout = nn.Dropout(0.8)
        self.seen = []

    def forward(self, unused, input_ids, sigma, labels):
        self.seen.append(input_ids.detach().clone())
        context = nn.functional.one_hot(input_ids, 11).float().mean(dim=1) @ self.context
        return self.dropout(self.position[None, :, :] + context[:, None, :])


def _model():
    return MDLM(ContextBackbone(), 11, 10, ModelConfig(length=8)).eval()


def _fixture():
    # Whole groups of two answers stay on each rank. Rank zero has 26 target
    # tokens; rank one has only 10, deliberately invalidating rank-local means.
    prefix_lengths = [1, 1, 2, 2, 5, 5, 6, 6]
    ids = (torch.arange(64).reshape(8, 8) % 9).long()
    for row, prefix_length in enumerate(prefix_lengths):
        ids[row, :prefix_length] = (torch.arange(prefix_length) + row // 2) % 9
    ids[:, -2] = 0  # EOS is still a target.
    ids[:, -1] = 9  # PAD is still a target.
    target = token_mask(prefix_lengths, 8, torch.device("cpu"))
    masks = [make_prompt_mask(target, 0.4, seed) for seed in (13, 20)]
    advantages = group_advantages(torch.tensor([1., 0., 0., 1., 1., 0., 1., 0.]), 2)
    return ids, target, masks, advantages


def _config(micro, accumulation):
    return {
        "micro_batch_size": micro, "gradient_accumulation_steps": accumulation,
        "precision": "fp32", "estimator_temperature": 1.0,
        "clip_epsilon": 0.2, "kl_beta": 0.04, "grad_clip": 1000.0,
    }


def _run_updates(world, rank, micro):
    model = _model()
    reference = copy.deepcopy(model).requires_grad_(False).eval()
    with torch.no_grad():
        reference.backbone.position.add_(torch.cos(torch.arange(88).float()).reshape(8, 11) * 0.25)
    scorer = PolicyScorer(model).eval()
    reference_scorer = PolicyScorer(reference).eval()
    wrapped = DDP(scorer, broadcast_buffers=False) if world > 1 else scorer
    ids, target, masks, advantages = _fixture()
    rows = slice(rank * 4, (rank + 1) * 4) if world > 1 else slice(None)
    ids, target, advantages = ids[rows], target[rows], advantages[rows]
    masks = [mask[rows] for mask in masks]
    config = _config(micro, (len(ids) + micro - 1) // micro)
    device = torch.device("cpu")
    old = cache_logps(scorer, ids, target, masks, config, device)
    ref = cache_logps(reference_scorer, ids, target, masks, config, device)
    old_snapshot = [value.clone() for value in old]
    optimizer = torch.optim.SGD(model.parameters(), lr=0.15)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    ctx = SimpleNamespace(world=world, rank=rank, device=device)
    metrics = []
    for inner in range(2):
        metrics.append(update_policy(
            wrapped, model, ids, target, masks[inner], old[inner], ref[inner], advantages,
            optimizer, scheduler, ctx, config))
    return {
        "state": {name: value.detach().cpu() for name, value in model.state_dict().items()},
        "gradients": {name: value.grad.detach().cpu() for name, value in model.named_parameters()},
        "metrics": metrics,
        "reference_has_grad": any(value.grad is not None for value in reference.parameters()),
        "old_unchanged": all(torch.equal(before, after) for before, after in zip(old_snapshot, old)),
        "local_target_tokens": int(target.sum()),
    }


def _distributed_worker(rank, port, output):
    torch.set_num_threads(1)
    # Explicitly disable libuv: some supported Windows CPU wheels omit it.
    store = dist.TCPStore("127.0.0.1", port, 2, rank == 0,
                          timeout=timedelta(seconds=60), use_libuv=False)
    dist.init_process_group("gloo", store=store, rank=rank, world_size=2,
                            timeout=timedelta(seconds=60))
    try:
        result = _run_updates(world=2, rank=rank, micro=2)
        torch.save(result, Path(output) / f"rank{rank}.pt")
    finally:
        dist.destroy_process_group()


class RLTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def assert_same_update(self, left, right):
        for field in ("state", "gradients"):
            self.assertEqual(set(left[field]), set(right[field]))
            for name in left[field]:
                with self.subTest(field=field, parameter=name):
                    torch.testing.assert_close(left[field][name], right[field][name], atol=2e-7, rtol=2e-6)
        for first, second in zip(left["metrics"], right["metrics"]):
            self.assertEqual(set(first), set(second))
            for key in first:
                with self.subTest(metric=key):
                    self.assertAlmostEqual(first[key], second[key], delta=2e-7)
        self.assertFalse(left["reference_has_grad"])
        self.assertFalse(right["reference_has_grad"])
        self.assertTrue(left["old_unchanged"])
        self.assertTrue(right["old_unchanged"])

    def test_accumulated_microbatches_match_single_full_batch(self):
        accumulated = _run_updates(world=1, rank=0, micro=2)
        full = _run_updates(world=1, rank=0, micro=8)
        self.assert_same_update(accumulated, full)
        self.assertGreater(full["metrics"][0]["kl"], 0)
        self.assertGreater(full["metrics"][0]["grad_norm_before_clip"], 0)
        self.assertEqual(full["metrics"][0]["mean_ratio"], 1.0)
        self.assertNotEqual(full["metrics"][1]["mean_ratio"], 1.0)

    def test_two_gloo_ranks_match_serial_global_token_average(self):
        self.assertTrue(dist.is_available() and dist.is_gloo_available(), "Actual Gloo coverage is required")
        with socket.socket() as socket_handle:
            socket_handle.bind(("127.0.0.1", 0))
            port = socket_handle.getsockname()[1]
        with tempfile.TemporaryDirectory(prefix="mdlm-rl-ddp-") as output:
            mp.spawn(_distributed_worker, args=(port, output), nprocs=2, join=True)
            ranks = [torch.load(Path(output) / f"rank{rank}.pt", weights_only=True) for rank in range(2)]
        serial = _run_updates(world=1, rank=0, micro=2)
        self.assertEqual([rank["local_target_tokens"] for rank in ranks], [26, 10])
        for rank in ranks:
            self.assert_same_update(rank, serial)
            self.assertEqual(rank["metrics"][0]["target_tokens"], 36)

    def test_cache_masks_whole_completion_and_is_detached_and_stable(self):
        model = _model()
        scorer = PolicyScorer(model).eval()
        ids, target, masks, _ = _fixture()
        clean = ids.clone()
        cached = cache_logps(scorer, ids, target, masks, _config(2, 4), torch.device("cpu"))
        self.assertEqual(len(cached), 2)
        self.assertTrue(all(not value.requires_grad and value.grad_fn is None for value in cached))
        self.assertTrue(all(value.device.type == "cpu" for value in cached))
        self.assertTrue(all(torch.all(value[target] < 0) for value in cached))
        self.assertTrue(all(torch.all(value[~target] == 0) for value in cached))
        self.assertTrue(torch.equal(ids, clean))
        for inner in range(2):
            seen = torch.cat(model.backbone.seen[inner * 4:(inner + 1) * 4])
            self.assertTrue(torch.equal(seen, ids.masked_fill(target | masks[inner], model.mask_index)))
        self.assertFalse(torch.equal(cached[0], cached[1]))
        copied = [value.clone() for value in cached]
        with torch.no_grad():
            model.backbone.context.add_(0.3)
        self.assertTrue(all(torch.equal(before, after) for before, after in zip(copied, cached)))

    def test_real_dit_eval_backward_connects_every_parameter(self):
        from mdlm_rl.transformer import DIT

        config = ModelConfig(hidden_size=32, cond_dim=16, n_blocks=1, n_heads=4, length=8, dropout=0.8)
        architecture = {"model": {**asdict(config), "softcap": 0.0},
                        "algo": {"causal_attention": False, "adaLN": True}}
        with torch.random.fork_rng():
            torch.manual_seed(7)
            model = MDLM(DIT(architecture, vocab_size=11), 11, 10, config).eval()
        scorer = PolicyScorer(model).eval()
        ids, target, masks, advantages = _fixture()
        current = scorer(ids, target, masks[0])
        torch.testing.assert_close(current, scorer(ids, target, masks[0]), atol=0, rtol=0)
        old = current.detach().clone().requires_grad_(True)
        reference = current.detach().clone().requires_grad_(True)
        result = policy_loss(current, old, reference, advantages, target, 0.2, 0.04)
        (result["loss_sum"] / result["target_tokens"]).backward()
        self.assertFalse(model.training)
        self.assertTrue(all(not module.training for module in model.modules()))
        self.assertIsNone(old.grad)
        self.assertIsNone(reference.grad)
        self.assertEqual([name for name, value in model.named_parameters() if value.grad is None], [])
        self.assertTrue(all(bool(torch.isfinite(value.grad).all()) for value in model.parameters()))
        self.assertGreater(sum(float(value.grad.abs().sum()) for value in model.parameters()), 0)


if __name__ == "__main__":
    unittest.main()
