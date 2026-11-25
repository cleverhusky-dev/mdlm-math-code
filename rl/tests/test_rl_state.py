"""Synthetic sampler and resume fixtures; no server or benchmark examples."""
import copy
import json
from pathlib import Path
import random
import tempfile
import unittest

import torch

from mdlm_rl.checkpoints import atomic_torch_save
from mdlm_rl.state import (
    SCHEMA, MODEL_SCHEMA, QuestionSampler, check_resume, load_rl,
    make_best_payload, resume_signature,
)


class QuestionSamplerTests(unittest.TestCase):
    def test_cross_epoch_batches_are_unique_and_every_epoch_is_complete(self):
        sampler = QuestionSampler(11, 42)
        stream = []
        for _ in range(22):
            batch = sampler.next_batch(8)
            self.assertEqual(len(set(batch)), 8)
            stream.extend(batch)
        for start in range(0, len(stream), 11):
            self.assertEqual(sorted(stream[start:start + 11]), list(range(11)))
        self.assertEqual(sampler.questions_seen, 176)

    def test_resume_after_crossing_epoch_matches_uninterrupted_stream(self):
        sampler = QuestionSampler(13, 1)
        for _ in range(3):
            sampler.next_batch(8)
        state = sampler.state_dict()
        resumed = QuestionSampler(13, 1, state)
        # Resume also works after a human-readable JSON round trip.
        from_json = QuestionSampler(13, 1, json.loads(json.dumps(state)))
        for _ in range(15):
            expected = sampler.next_batch(8)
            self.assertEqual(resumed.next_batch(8), expected)
            self.assertEqual(from_json.next_batch(8), expected)
        self.assertEqual(resumed.state_dict(), sampler.state_dict())

    def test_private_random_state_and_returned_state_do_not_mutate_sampler(self):
        random.seed(123)
        before = random.getstate()
        sampler = QuestionSampler(10, 123)
        sampler.next_batch(8)
        self.assertEqual(before, random.getstate())
        checkpoint = sampler.state_dict()
        original_order = sampler.order.copy()
        checkpoint["order"].reverse()
        self.assertEqual(sampler.order, original_order)

    def test_exact_division_and_single_question_dataset(self):
        sampler = QuestionSampler(8, 4)
        for _ in range(4):
            self.assertEqual(sorted(sampler.next_batch(8)), list(range(8)))
        single = QuestionSampler(1, 4)
        self.assertEqual([single.next_batch(1) for _ in range(3)], [[0], [0], [0]])
        self.assertEqual(single.questions_seen, 3)

    def test_invalid_arguments_and_state_are_rejected(self):
        for size, seed in ((0, 1), (True, 1), (3, -1), (3, False)):
            with self.subTest(size=size, seed=seed), self.assertRaises(ValueError):
                QuestionSampler(size, seed)
        sampler = QuestionSampler(10, 1)
        for n in (0, 11, True, 1.5):
            with self.subTest(n=n), self.assertRaises(ValueError):
                sampler.next_batch(n)
        valid = sampler.state_dict()
        mutations = {
            "size": 11, "seed": 2, "epoch": -1, "cursor": 11,
            "order": [0] * 10, "rng_state": [0], "schema": "wrong",
        }
        for key, value in mutations.items():
            with self.subTest(key=key), self.assertRaises(ValueError):
                QuestionSampler(10, 1, {**valid, key: value})


class RLCheckpointTests(unittest.TestCase):
    def setUp(self):
        self.config = {
            "max_steps": 1000, "max_wall_seconds": 86400,
            "output_dir": "runs/rl/experiment1", "learning_rate": 3e-6,
            "initial_checkpoint": "runs/experiment12/training/checkpoints/best.pt",
            "rollout": {"temperature": 0.1, "steps": 128, "inner_updates": 4},
        }
        self.identity = {"world_size": 2, "data_sha256": "data", "source_sha256": "code"}
        self.payload = {
            "schema": SCHEMA,
            "model": {"weight": torch.tensor([1.0, 2.0])},
            "model_config": {"length": 512},
            "optimizer": {"state": {0: {"step": torch.tensor(100)}}},
            "scheduler": {"last_epoch": 100},
            "progress": {"global_step": 100, "best_step": 100, "best_correct": 10},
            "rng_by_rank": [{"rank": 0}, {"rank": 1}],
            "sampler": QuestionSampler(100, 1).state_dict(),
            "config": self.config,
            "identity": self.identity,
        }

    def test_full_checkpoint_round_trip_and_allowed_bound_extension(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "last.pt"
            atomic_torch_save(self.payload, path)
            loaded = load_rl(path)
            self.assertTrue(torch.equal(loaded["model"]["weight"], self.payload["model"]["weight"]))
            changed = {**self.config, "max_steps": 2000, "max_wall_seconds": 172800,
                       "output_dir": "runs/rl/continued"}
            check_resume(loaded, changed, self.identity)
            self.assertEqual(resume_signature(self.config), resume_signature(changed))

    def test_resume_rejects_algorithm_input_and_identity_changes(self):
        for key, value in (("learning_rate", 1e-5), ("initial_checkpoint", "other.pt"),
                           ("rollout", {**self.config["rollout"], "temperature": 1.0})):
            with self.subTest(key=key), self.assertRaises(ValueError):
                check_resume(self.payload, {**self.config, key: value}, self.identity)
        for key, value in (("world_size", 1), ("data_sha256", "other"), ("source_sha256", "other")):
            with self.subTest(identity=key), self.assertRaises(ValueError):
                check_resume(self.payload, self.config, {**self.identity, key: value})
        with self.assertRaises(ValueError):
            check_resume(self.payload, {**self.config, "max_steps": 99}, self.identity)

    def test_incomplete_wrong_schema_and_rng_count_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "bad.pt"
            for key in ("optimizer", "sampler", "rng_by_rank", "identity"):
                broken = {k: v for k, v in self.payload.items() if k != key}
                atomic_torch_save(broken, path)
                with self.subTest(missing=key), self.assertRaises(ValueError):
                    load_rl(path)
            for change in ({"schema": "mdlm-sft-v1"},
                           {"rng_by_rank": [{"rank": 0}]},
                           {"progress": {"global_step": -1}}):
                atomic_torch_save({**self.payload, **change}, path)
                with self.subTest(change=change), self.assertRaises(ValueError):
                    load_rl(path)

    def test_best_is_lightweight_metadata_snapshot_and_not_resumable(self):
        best = make_best_payload(self.payload)
        self.assertEqual(best["schema"], MODEL_SCHEMA)
        self.assertNotIn("optimizer", best)
        self.assertNotIn("scheduler", best)
        self.assertNotIn("rng_by_rank", best)
        self.assertEqual(best["model_config"], {"length": 512})
        self.assertIs(best["model"], self.payload["model"])
        self.payload["progress"]["global_step"] = 200
        self.payload["config"]["rollout"]["steps"] = 32
        self.assertEqual(best["progress"]["global_step"], 100)
        self.assertEqual(best["config"]["rollout"]["steps"], 128)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "best.pt"
            atomic_torch_save(best, path)
            with self.assertRaises(ValueError):
                load_rl(path)

    def test_best_model_config_falls_back_to_identity(self):
        del self.payload["model_config"]
        self.payload["identity"]["model"] = {"length": 512, "hidden_size": 768}
        best = make_best_payload(self.payload)
        self.assertEqual(best["model_config"], self.payload["identity"]["model"])
        self.payload["identity"]["model"]["length"] = 256
        self.assertEqual(best["model_config"]["length"], 512)


if __name__ == "__main__":
    unittest.main()
