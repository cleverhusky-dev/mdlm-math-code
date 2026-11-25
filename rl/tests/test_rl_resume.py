"""Trainer transactions on tiny CPU models, with no benchmark or executor access.

Generation and execution are deterministic fixtures; policy gradients, Adam,
scheduling, RNG capture, journal recovery and checkpoint publication are real.
"""
from contextlib import redirect_stdout
import copy
import io
import json
import os
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from mdlm_rl import train
from mdlm_rl.diffusion import MDLM, ModelConfig
from mdlm_rl.distributed import DistributedContext
from mdlm_rl.runtime import capture_rng, isolated_rng, seed_all
from mdlm_rl.state import load_rl


class FixtureBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.logits = nn.Parameter(torch.linspace(-0.4, 0.4, 30).reshape(6, 5))

    def forward(self, unused, input_ids, sigma, labels):
        return self.logits.unsqueeze(0).expand(len(input_ids), -1, -1)


class FixtureTokenizer:
    def decode(self, tokens, **kwargs):
        return " ".join(map(str, tokens))


def fixture_prompt(question, tokenizer, length):
    prefix = [0, 1] + ([2] if int(question) % 2 else [])
    return {"input_ids": prefix + [3] * (length - len(prefix)),
            "prefix_length": len(prefix)}


class RLResumeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="rl-resume-fixture-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        environment = patch.dict(os.environ, {
            "RANK": "0", "WORLD_SIZE": "1", "LOCAL_RANK": "0", "LOCAL_WORLD_SIZE": "1",
        })
        environment.start()
        self.addCleanup(environment.stop)
        self.ctx = DistributedContext("cpu")
        self.addCleanup(self.ctx.close)
        self.generation_calls = 0
        self.fail_generation_call = None
        self.fail_validation_steps = set()
        self.validation_counts = {}
        self.validation_calls = []
        self.validation_weights = {}
        self.events = []
        for target, replacement in (
            ("encode_prompt", fixture_prompt),
            ("ancestral_sample", self.generate),
            ("score_program", self.score),
            ("generate_and_score", self.evaluate),
        ):
            mock = patch.object(train, target, side_effect=replacement)
            mock.start()
            self.addCleanup(mock.stop)

    def generate(self, model, prefixes, *, length, **kwargs):
        self.generation_calls += 1
        self.events.append("generate")
        # Consume all three RNG streams; recovery must restore each one.
        random.random()
        np.random.random()
        ids = torch.randint(0, 4, (len(prefixes), length))
        if self.generation_calls == self.fail_generation_call:
            raise TimeoutError("fixture generation interrupted after consuming RNG")
        for index, prefix in enumerate(prefixes):
            ids[index, :len(prefix)] = torch.tensor(prefix)
            # Each group has one correct and one wrong answer, hence a real
            # nonzero policy gradient instead of an all-zero reward smoke test.
            ids[index, len(prefix)] = index % 2
        return ids

    @staticmethod
    def score(text, gold, **kwargs):
        return {"status": "correct" if text.split()[0] == "0" else "wrong_answer"}

    def evaluate(self, model, tokenizer, questions, settings, device, precision,
                 output_dir, *, checkpoint_source, cancel_requested, **kwargs):
        step = checkpoint_source["rl_global_step"]
        self.events.append(f"validate:{step}")
        self.validation_calls.append(step)
        self.validation_weights[step] = copy.deepcopy(model.state_dict())
        Path(output_dir).mkdir(parents=True)
        # The production evaluator uses isolated_rng, including on exceptions.
        with isolated_rng(settings["seed"]):
            random.random()
            np.random.random()
            torch.rand(3)
            if step in self.fail_validation_steps or cancel_requested():
                self.fail_validation_steps.discard(step)
                raise TimeoutError("fixture validation interrupted")
        return {"complete": True, "question_count": len(questions),
                "counts": {"correct": self.validation_counts.get(step, 1)}}

    def build(self, name, *, max_steps=8, resume=None):
        output = self.root / name
        output.mkdir(exist_ok=True)
        config = json.loads((Path(__file__).resolve().parents[1] /
                             "experiments/experiment1/config.json").read_text(encoding="utf-8"))
        config.update(output_dir=str(output), seed=27, world_size=1, precision="fp32",
                      expected_train_questions=3, expected_val_questions=2,
                      questions_per_batch=2, samples_per_question=2, inner_updates=4,
                      micro_batch_size=2, gradient_accumulation_steps=2,
                      generation_batch_size=2, learning_rate=0.03, warmup_steps=2,
                      max_steps=max_steps, max_wall_seconds=3600,
                      validation_every_steps=4, checkpoint_every_steps=4)
        config["rollout"]["length"] = config["evaluation"]["length"] = 6
        config["rollout"]["steps"] = config["evaluation"]["steps"] = 2
        model = MDLM(FixtureBackbone(), vocab_size=5, mask_index=4,
                     config=ModelConfig(length=6))
        reference = copy.deepcopy(model)
        rows = [{"gsm8k_id": f"train:{index}", "question": str(index), "gold_answer": 1}
                for index in range(5)]
        identity = {"world_size": 1, "initial_checkpoint_sha256": "a" * 64,
                    "model": {"fixture": "position-logits", "length": 6},
                    "code": "fixture-source", "data": "fixture-questions"}
        return train.Trainer(config, self.ctx, model, reference, FixtureTokenizer(),
                             rows[:3], rows[3:], identity, resume=resume)

    def run_quietly(self, trainer):
        with redirect_stdout(io.StringIO()):
            return trainer.run()

    def load(self, name):
        return load_rl(self.root / name / "checkpoints/last.pt")

    def assert_nested_equal(self, expected, actual):
        if isinstance(expected, torch.Tensor):
            self.assertTrue(torch.equal(expected, actual), "Tensor differs after resume")
        elif isinstance(expected, np.ndarray):
            np.testing.assert_array_equal(expected, actual)
        elif isinstance(expected, dict):
            self.assertEqual(expected.keys(), actual.keys())
            for key in expected:
                self.assert_nested_equal(expected[key], actual[key])
        elif isinstance(expected, (tuple, list)):
            self.assertEqual(len(expected), len(actual))
            for left, right in zip(expected, actual):
                self.assert_nested_equal(left, right)
        else:
            self.assertEqual(expected, actual)

    def assert_training_equal(self, expected, actual):
        for key in ("model", "optimizer", "scheduler", "progress", "sampler", "rng_by_rank"):
            with self.subTest(state=key):
                self.assert_nested_equal(expected[key], actual[key])

    def baseline(self):
        seed_all(1234)
        trainer = self.build("baseline")
        self.assertEqual(self.run_quietly(trainer), "complete")
        return self.load("baseline")

    def test_uninterrupted_matches_extended_resume_weights_optimizer_and_all_rng(self):
        expected = self.baseline()
        seed_all(1234)
        first = self.build("resumed", max_steps=4)
        original = copy.deepcopy(first.model.state_dict())
        self.assertEqual(self.run_quietly(first), "complete")
        checkpoint = self.load("resumed")
        self.assertEqual(checkpoint["progress"]["global_step"], 4)
        seed_all(999)  # Resume must override unrelated process RNG state.
        resumed = self.build("resumed", resume=checkpoint)
        self.assertEqual(self.run_quietly(resumed), "complete")
        self.assert_training_equal(expected, self.load("resumed"))
        self.assert_nested_equal(original, resumed.reference.state_dict())
        self.assertFalse(torch.equal(original["backbone.logits"],
                                     resumed.model.state_dict()["backbone.logits"]))
        rows = [json.loads(row) for row in (resumed.output / "train.jsonl").read_text().splitlines()]
        self.assertEqual([row["step"] for row in rows], list(range(1, 9)))
        self.assertEqual(resumed.sampler.questions_seen, 4)  # Crossed a 3-question epoch.

    def test_step_zero_is_best_on_ties_and_first_later_improvement_is_retained(self):
        for counts, expected_step in (({}, 0), ({4: 2, 8: 2}, 4)):
            with self.subTest(best_step=expected_step):
                self.validation_counts = counts
                trainer = self.build(f"best-{expected_step}")
                self.assertEqual(self.run_quietly(trainer), "complete")
                directory = trainer.output / "checkpoints"
                best = torch.load(directory / "best.pt", weights_only=False)
                metadata = json.loads((directory / "best.json").read_text())
                self.assertEqual(best["progress"]["best_step"], expected_step)
                self.assertEqual(metadata["step"], expected_step)
                self.assertEqual(best["progress"]["global_step"], expected_step)
                self.assert_nested_equal(best["model"], self.validation_weights[expected_step])
                self.assertNotIn("optimizer", best)
                self.assertEqual(self.load(f"best-{expected_step}")["progress"]["global_step"], 8)

    def test_cancel_during_inner_update_commits_only_the_complete_rollout(self):
        expected = self.baseline()
        seed_all(1234)
        trainer = self.build("cancelled")
        update = train.update_policy
        calls = 0

        def cancel_after_first(*args, **kwargs):
            nonlocal calls
            result = update(*args, **kwargs)
            calls += 1
            if calls == 1:
                trainer.cancelled = True
            return result

        with patch.object(train, "update_policy", side_effect=cancel_after_first):
            self.assertEqual(self.run_quietly(trainer), "stopped")
        self.assertEqual(calls, 4)
        checkpoint = self.load("cancelled")
        self.assertEqual(checkpoint["progress"]["global_step"], 4)
        self.assertEqual(checkpoint["progress"]["rollout_batches"], 1)
        self.assertEqual(checkpoint["scheduler"]["last_epoch"], 4)
        self.assertTrue(checkpoint["progress"]["validation_pending"])
        resumed = self.build("cancelled", resume=checkpoint)
        self.assertEqual(self.run_quietly(resumed), "complete")
        self.assert_training_equal(expected, self.load("cancelled"))

    def test_pending_validation_is_retried_before_another_rollout(self):
        expected = self.baseline()
        seed_all(1234)
        trainer = self.build("pending")
        self.fail_validation_steps.add(4)
        self.assertEqual(self.run_quietly(trainer), "stopped")
        checkpoint = self.load("pending")
        self.assertTrue(checkpoint["progress"]["validation_pending"])
        self.assertEqual(checkpoint["progress"]["validated_step"], 0)
        self.assertEqual(checkpoint["progress"]["best_step"], 0)
        self.events.clear()
        resumed = self.build("pending", resume=checkpoint)
        self.assertEqual(self.run_quietly(resumed), "complete")
        self.assertEqual(self.events[0], "validate:4")
        self.assert_training_equal(expected, self.load("pending"))
        attempts = list((trainer.output / "validation/step-00000004").glob("attempt-*"))
        self.assertEqual(len(attempts), 2)

    def test_initial_validation_interruption_does_not_select_partial_best(self):
        trainer = self.build("initial", max_steps=4)
        self.fail_validation_steps.add(0)
        self.assertEqual(self.run_quietly(trainer), "stopped")
        checkpoint = self.load("initial")
        self.assertIsNone(checkpoint["progress"]["best_step"])
        self.assertEqual(checkpoint["progress"]["global_step"], 0)
        self.assertFalse((trainer.output / "checkpoints/best.pt").exists())
        self.events.clear()
        resumed = self.build("initial", max_steps=4, resume=checkpoint)
        self.assertEqual(self.run_quietly(resumed), "complete")
        self.assertEqual(self.events[0], "validate:0")
        self.assertEqual(self.load("initial")["progress"]["best_step"], 0)

    def test_interrupted_generation_restores_sampler_and_rng_before_retry(self):
        expected = self.baseline()
        seed_all(1234)
        trainer = self.build("generation")
        before_rng = capture_rng()
        before_sampler = trainer.sampler.state_dict()
        # Interrupt the second microbatch after the first has already generated
        # and scored records; none may become an optimizer transaction.
        self.fail_generation_call = self.generation_calls + 2
        self.assertEqual(self.run_quietly(trainer), "stopped")
        checkpoint = self.load("generation")
        self.assertEqual(checkpoint["progress"]["global_step"], 0)
        self.assert_nested_equal(before_sampler, checkpoint["sampler"])
        self.assert_nested_equal(before_rng, checkpoint["rng_by_rank"][0])
        self.assertFalse((trainer.output / "rollouts").exists())
        self.fail_generation_call = None
        resumed = self.build("generation", resume=checkpoint)
        self.assertEqual(self.run_quietly(resumed), "complete")
        self.assert_training_equal(expected, self.load("generation"))

    def test_optimizer_timeout_keeps_old_checkpoint_and_reconciles_rolled_back_log(self):
        expected = self.baseline()
        seed_all(1234)
        trainer = self.build("optimizer-failure")
        update = train.update_policy
        calls = 0

        def fail_second(*args, **kwargs):
            nonlocal calls
            result = update(*args, **kwargs)
            calls += 1
            if calls == 2:
                raise TimeoutError("fixture optimizer transaction interrupted")
            return result

        with patch.object(train, "update_policy", side_effect=fail_second):
            with self.assertRaisesRegex(TimeoutError, "optimizer transaction"):
                self.run_quietly(trainer)
        checkpoint = self.load("optimizer-failure")
        self.assertEqual(checkpoint["progress"]["global_step"], 0)
        self.assertEqual(trainer.progress["global_step"], 1)
        resumed = self.build("optimizer-failure", resume=checkpoint)
        self.assertEqual(self.run_quietly(resumed), "complete")
        self.assert_training_equal(expected, self.load("optimizer-failure"))
        histories = list((trainer.output / "log_history").glob("train-*.jsonl"))
        self.assertEqual(len(histories), 1)
        self.assertEqual(json.loads(histories[0].read_text().strip())["step"], 1)

    def test_log_storage_timeout_cannot_publish_a_partial_inner_update(self):
        trainer = self.build("storage-failure")
        append = train.append_durable_jsonl

        def write_then_timeout(path, record, **kwargs):
            append(path, record, **kwargs)
            if Path(path).name == "train.jsonl" and record["step"] == 1:
                raise TimeoutError("fixture storage timed out after first optimizer update")

        with patch.object(train, "append_durable_jsonl", side_effect=write_then_timeout):
            with self.assertRaisesRegex(TimeoutError, "fixture storage"):
                self.run_quietly(trainer)
        checkpoint = self.load("storage-failure")
        self.assertEqual(trainer.progress["global_step"], 1)
        self.assertEqual(checkpoint["progress"]["global_step"], 0)
        self.assertEqual(checkpoint["progress"]["rollout_batches"], 0)
        self.assertEqual(checkpoint["scheduler"]["last_epoch"], 0)
        self.assertEqual(checkpoint["sampler"]["cursor"], 0)
        self.assertEqual(json.loads((trainer.output / "status.json").read_text())["status"], "failed")

    def test_incomplete_validation_result_cannot_select_best(self):
        trainer = self.build("partial-validation")
        partial = {"complete": False, "question_count": 1, "counts": {"correct": 1}}
        with patch.object(train, "generate_and_score", return_value=partial):
            with self.assertRaisesRegex(ValueError, "Incomplete validation"):
                self.run_quietly(trainer)
        checkpoint = self.load("partial-validation")
        self.assertIsNone(checkpoint["progress"]["best_step"])
        self.assertTrue(checkpoint["progress"]["validation_pending"])
        self.assertFalse((trainer.output / "checkpoints/best.pt").exists())

    def test_resume_rejects_inconsistent_optimizer_rollout_and_sampler_cursors(self):
        trainer = self.build("invalid", max_steps=4)
        self.assertEqual(self.run_quietly(trainer), "complete")
        valid = self.load("invalid")
        for field in ("global_step", "rollout_batches", "scheduler", "sampler"):
            invalid = copy.deepcopy(valid)
            if field in ("global_step", "rollout_batches"):
                invalid["progress"][field] += 1
            elif field == "scheduler":
                invalid["scheduler"]["last_epoch"] -= 1
            else:
                invalid["sampler"]["cursor"] -= 1
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "cursors disagree"):
                self.build("invalid", resume=invalid)


if __name__ == "__main__":
    unittest.main()
