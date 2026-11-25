"""CPU fixtures for temperature semantics; never read official test questions."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn

from mdlm_sft.diffusion import MDLM, ModelConfig, subs_log_probs, validate_temperature
from mdlm_sft.evaluation import generate_and_score
from mdlm_sft.sampling import ancestral_sample


def legacy_subs(logits, xt, mask_index):
    """Frozen arithmetic from the temperature=1 implementation before this change."""
    logits = logits.float().clone()
    logits[..., mask_index] += -1_000_000.0
    log_probs = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
    visible = xt != mask_index
    log_probs[visible] = -1_000_000.0
    log_probs[visible, xt[visible]] = 0
    return log_probs


class FixedBackbone(nn.Module):
    def __init__(self, values=(0.25, 0.5, -0.25, 0.125), dtype=torch.float32):
        super().__init__()
        self.values = nn.Parameter(torch.tensor(values, dtype=dtype))

    def forward(self, unused, xt, sigma, labels):
        return self.values.expand(*xt.shape, len(self.values))


def fixture_model(length=12, values=(0.25, 0.5, -0.25, 0.125), dtype=torch.float32):
    return MDLM(FixedBackbone(values, dtype), len(values), len(values) - 1,
                ModelConfig(length=length))


@torch.no_grad()
def legacy_sample(model, prefixes, steps):
    """Frozen previous ancestral loop, including random draw shape and order."""
    xt = torch.full((len(prefixes), model.num_tokens), model.mask_index, dtype=torch.long)
    for i, prefix in enumerate(prefixes):
        xt[i, :len(prefix)] = torch.tensor(prefix)
    start = min(map(len, prefixes))
    times = torch.linspace(1, 1e-5, steps + 1)
    ones = torch.ones(len(prefixes), 1, dtype=model.dtype)
    for index in range(steps):
        _, alpha_t = model.noise(times[index] * ones)
        _, alpha_s = model.noise(times[index + 1] * ones)
        final = index == steps - 1
        if final:
            alpha_s = torch.ones_like(alpha_t)
        logits = model.backbone(None, xt, torch.zeros(len(prefixes)), None)
        probs = legacy_subs(logits, xt, model.mask_index).to(torch.float64)[:, start:].exp()
        noise = 1e-10 - (torch.rand_like(probs) + 1e-10).log()
        sampled = (probs / noise).argmax(dim=-1)
        should_denoise = (torch.ones_like(sampled, dtype=torch.bool) if final else
                          torch.rand_like(sampled, dtype=torch.float64) < (alpha_s - alpha_t) / (1 - alpha_t))
        xt[:, start:] = torch.where((xt[:, start:] == model.mask_index) & should_denoise,
                                   sampled, xt[:, start:])
        for i, prefix in enumerate(prefixes):
            xt[i, :len(prefix)] = torch.tensor(prefix)
    return xt


class TemperatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_temperature_one_log_probs_are_bitwise_legacy_for_fp32_and_bf16(self):
        xt = torch.tensor([[3, 0, 3], [1, 3, 2]])
        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype):
                model = fixture_model(dtype=dtype)
                logits = model.backbone(None, xt, None, None)
                expected = legacy_subs(logits, xt, 3)
                self.assertTrue(torch.equal(subs_log_probs(logits, xt, 3), expected))
                self.assertTrue(torch.equal(model.log_probs(xt, temperature=1), expected))

    def test_temperature_one_sampling_and_rng_are_bitwise_legacy(self):
        prefixes = [[0], [1, 2, 0]]
        for dtype in (torch.float32, torch.bfloat16):
            for steps in (1, 32, 128):
                with self.subTest(dtype=dtype, steps=steps):
                    model = fixture_model(dtype=dtype)
                    torch.manual_seed(41)
                    expected = legacy_sample(model, prefixes, steps)
                    expected_rng = torch.get_rng_state()
                    torch.manual_seed(41)
                    actual = ancestral_sample(model, prefixes, steps=steps, length=12, temperature=1)
                    self.assertTrue(torch.equal(actual, expected))
                    self.assertTrue(torch.equal(torch.get_rng_state(), expected_rng))

    def test_low_temperature_matches_logits_divided_by_temperature(self):
        logits = torch.tensor([[[0.25, 0.5, -0.25, 100.0]]], dtype=torch.bfloat16)
        actual = subs_log_probs(logits, torch.tensor([[3]]), 3, temperature=0.1).exp()
        expected = torch.softmax(logits.float()[..., :3] / 0.1, dim=-1)
        torch.testing.assert_close(actual[..., :3], expected, atol=1e-7, rtol=1e-6)
        self.assertEqual(actual[0, 0, 3].item(), 0)
        model = fixture_model(values=(0.25, 0.5, -0.25, 100.0), dtype=torch.bfloat16)
        self.assertTrue(torch.equal(model.log_probs(torch.tensor([[3]]), temperature=0.1).exp(), actual))

    def test_mask_stays_forbidden_and_visible_tokens_are_copied(self):
        logits = torch.tensor([[[2.0, 1.0, 0.0, 1e20], [0.0, 1.0, 2.0, 1e20]]])
        xt = torch.tensor([[3, 1]])
        for temperature in (0.1, 2.0, 1e8, 1e-300):
            with self.subTest(temperature=temperature):
                log_probs = subs_log_probs(logits, xt, 3, temperature=temperature)
                probs = log_probs.exp()
                self.assertFalse(torch.isnan(probs).any())
                torch.testing.assert_close(probs.sum(-1), torch.ones_like(xt, dtype=torch.float32))
                self.assertEqual(probs[0, 0, 3].item(), 0)
                self.assertEqual(log_probs[0, 1].tolist(), [-1e6, 0, -1e6, -1e6])

    def test_invalid_temperatures_reject_before_sampling(self):
        model = fixture_model()
        for value in (0, -0.1, float("nan"), float("inf"), -float("inf"), True, "0.1", None):
            with self.subTest(temperature=value):
                with self.assertRaisesRegex(ValueError, "positive finite"):
                    validate_temperature(value)
                with self.assertRaisesRegex(ValueError, "positive finite"):
                    model.log_probs(torch.tensor([[3]]), temperature=value)
                with self.assertRaisesRegex(ValueError, "positive finite"):
                    ancestral_sample(model, [[0]], length=12, temperature=value)

    def test_low_temperature_preserves_absorption_schedule_and_prefixes(self):
        prefixes = [[0], [1, 2, 0]]
        histories = []
        for temperature in (1.0, 0.1):
            model = fixture_model(length=32)
            history = []
            original = model.log_probs
            def observed(xt, *, temperature):
                history.append(xt.eq(model.mask_index).clone())
                return original(xt, temperature=temperature)
            torch.manual_seed(17)
            with patch.object(model, "log_probs", side_effect=observed) as calls:
                output = ancestral_sample(model, prefixes, length=32, steps=32, temperature=temperature)
                self.assertEqual(calls.call_count, 32)
                self.assertTrue(all(call.kwargs["temperature"] == temperature for call in calls.call_args_list))
            for row, prefix in zip(output, prefixes):
                self.assertEqual(row[:len(prefix)].tolist(), prefix)
            self.assertFalse(output.eq(model.mask_index).any())
            histories.append(history)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(*histories)))

    def test_final_transition_remains_stochastic_at_low_temperature(self):
        model = fixture_model(length=64, values=(0.0, 0.0, 0.0))
        torch.manual_seed(6)
        with patch.object(model, "log_probs", wraps=model.log_probs) as calls:
            output = ancestral_sample(model, [[0]], length=64, steps=1, temperature=0.1)
        self.assertEqual(calls.call_count, 1)
        self.assertEqual(set(output[0, 1:].tolist()), {0, 1})

    def test_training_loss_keeps_temperature_one_objective(self):
        model = fixture_model(length=4)
        inputs = torch.tensor([[0, 1, 2, 0]])
        target = torch.tensor([[False, True, True, True]])
        t = torch.tensor([0.5])
        uniforms = torch.tensor([[0.0, 0.0, 0.75, 0.0]])
        result = model(inputs, target, t=t, corruption_uniform=uniforms)
        xt, masked = model.corrupt(inputs, target, t, uniforms)
        log_probs = legacy_subs(model.backbone(None, xt, None, None), xt, model.mask_index)
        expected = (-log_probs.gather(-1, inputs.unsqueeze(-1)).squeeze(-1) * masked / t[:, None]).sum(dtype=torch.float32)
        self.assertTrue(torch.equal(result["loss_sum"], expected))
        self.assertTrue(torch.equal(result["loss"], expected / target.sum()))

    def test_evaluation_passes_temperature_all_the_way_to_the_model(self):
        model = fixture_model(length=512)
        settings = {"length": 512, "temperature": 0.1, "use_float64": True, "steps": 3,
                    "batch_size": 2, "seed": 5, "timeout_s": 15, "memory_mb": 512}
        questions = [{"gsm8k_id": "fixture:0", "question": "A synthetic question.", "gold_answer": 1}]
        class Tokenizer:
            def decode(self, *args, **kwargs):
                return "synthetic generated fixture"
        with tempfile.TemporaryDirectory() as directory, \
                patch("mdlm_sft.evaluation.protocol_record", return_value={"protocol_sha256": "fixture", "generation": settings}), \
                patch("mdlm_sft.evaluation.encode_prompt", return_value={"input_ids": [0, 1], "prefix_length": 2}), \
                patch("mdlm_sft.scoring.preflight_executor", return_value={"status": "fixture"}), \
                patch("mdlm_sft.scoring.score_program", return_value={"status": "correct"}), \
                patch.object(model, "log_probs", wraps=model.log_probs) as calls, contextlib.redirect_stdout(io.StringIO()):
            summary = generate_and_score(model, Tokenizer(), questions, settings, torch.device("cpu"), "fp32",
                                         directory, checkpoint_source={"source": "fixture"}, label="fixture")
            self.assertTrue(summary["complete"])
            self.assertEqual(calls.call_count, 3)
            self.assertTrue(all(call.kwargs["temperature"] == 0.1 for call in calls.call_args_list))
            self.assertEqual(json.loads((Path(directory) / "protocol.json").read_text())["generation"]["temperature"], 0.1)
            self.assertTrue(model.training)


if __name__ == "__main__":
    unittest.main()
