"""Tiny CPU fixtures for diffusion policy gradients; no benchmark data access."""
import math
import unittest

import torch
from torch import nn

from mdlm_rl.diffusion import MDLM, ModelConfig
from mdlm_rl.objective import (
    PolicyScorer, group_advantages, make_prompt_mask, policy_loss,
)


class PositionBackbone(nn.Module):
    def __init__(self, length=5, vocab=5):
        super().__init__()
        self.logits = nn.Parameter(torch.linspace(-0.4, 0.4, length * vocab).reshape(length, vocab))
        self.last_input = None

    def forward(self, unused, input_ids, sigma, labels):
        self.last_input = input_ids.detach().clone()
        return self.logits.unsqueeze(0).expand(input_ids.shape[0], -1, -1)


def tiny_scorer():
    model = MDLM(PositionBackbone(), vocab_size=5, mask_index=4, config=ModelConfig(length=5))
    return PolicyScorer(model).eval()


def token_fixture():
    ids = torch.tensor([[0, 1, 2, 0, 3], [0, 2, 3, 1, 0]])
    targets = torch.tensor([[False, False, True, True, True], [False, True, True, True, True]])
    return ids, targets


class RLObjectiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def loss(self, current, old, advantages, mask=None, ref=None, beta=0.0):
        if mask is None:
            mask = torch.ones_like(current, dtype=torch.bool)
        return policy_loss(current, old, old if ref is None else ref,
                           advantages, mask, clip_epsilon=0.2, kl_beta=beta)

    def test_prompt_mask_reproducible_private_and_completion_excluded(self):
        targets = torch.zeros((16, 32), dtype=torch.bool)
        targets[:, 20:] = True
        before = torch.get_rng_state().clone()
        first = make_prompt_mask(targets, 0.3, 19)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertTrue(torch.equal(first, make_prompt_mask(targets, 0.3, 19)))
        self.assertFalse(torch.equal(first, make_prompt_mask(targets, 0.3, 20)))
        self.assertFalse(bool((first & targets).any()))
        self.assertTrue(torch.equal(make_prompt_mask(targets, 1, 5), ~targets))
        self.assertFalse(bool(make_prompt_mask(targets, 0, 5).any()))

    def test_prompt_mask_rejects_invalid_probability_seed_and_mask(self):
        _, target = token_fixture()
        for p in (True, -0.1, 1.1, float("nan"), float("inf")):
            with self.subTest(p=p), self.assertRaises(ValueError):
                make_prompt_mask(target, p, 1)
        for seed in (True, -1, 2**63, 1.5):
            with self.subTest(seed=seed), self.assertRaises(ValueError):
                make_prompt_mask(target, 0.15, seed)
        with self.assertRaises(ValueError):
            make_prompt_mask(target.float(), 0.15, 1)

    def test_scorer_masks_every_completion_without_self_copy_shortcut(self):
        scorer = tiny_scorer()
        ids, target = token_fixture()
        prompt = make_prompt_mask(target, 0, 1)
        result = scorer(ids, target, prompt)
        seen = scorer.model.backbone.last_input
        self.assertTrue(torch.equal(seen[target], torch.full_like(ids[target], 4)))
        self.assertTrue(torch.equal(seen[~target], ids[~target]))
        self.assertTrue(bool((result[target] < 0).all()))
        self.assertTrue(bool((result[~target] == 0).all()))
        # The underlying SUBS wrapper would give exact copying probability 1
        # if a caller mistakenly scored the visible generated answer.
        copied = scorer.model.log_probs(ids).gather(-1, ids.unsqueeze(-1)).squeeze(-1)
        self.assertTrue(bool((copied[target] == 0).all()))

    def test_scorer_all_prompt_tokens_eligible_and_targets_unchanged(self):
        scorer = tiny_scorer()
        ids, target = token_fixture()
        clean = ids.clone()
        logps = scorer(ids, target, make_prompt_mask(target, 1, 11))
        self.assertTrue(bool((scorer.model.backbone.last_input == 4).all()))
        self.assertTrue(torch.equal(clean, ids))
        self.assertTrue(bool(torch.isfinite(logps).all()))

    def test_scorer_eval_mode_keeps_current_parameter_gradients(self):
        scorer = tiny_scorer()
        ids, target = token_fixture()
        logps = scorer(ids, target, make_prompt_mask(target, 0.15, 1))
        result = self.loss(logps, logps.detach(), torch.ones(2), target)
        self.assertTrue(result["loss_sum"].requires_grad)
        result["loss_sum"].backward()
        self.assertIsNotNone(scorer.model.backbone.logits.grad)
        self.assertGreater(float(scorer.model.backbone.logits.grad.abs().sum()), 0)
        self.assertTrue(bool((scorer.model.backbone.logits.grad[0] == 0).all()))

    def test_scorer_temperature_one_is_explicit_untempered_surrogate(self):
        scorer = tiny_scorer()
        ids, target = token_fixture()
        prompt = make_prompt_mask(target, 0.15, 2)
        self.assertTrue(torch.equal(scorer(ids, target, prompt),
                                    scorer(ids, target, prompt, temperature=1.0)))
        self.assertFalse(torch.equal(scorer(ids, target, prompt),
                                     scorer(ids, target, prompt, temperature=0.1)))

    def test_scorer_rejects_overlapping_masks_invalid_tokens_and_non_suffix(self):
        scorer = tiny_scorer()
        ids, target = token_fixture()
        with self.assertRaises(ValueError):
            scorer(ids, target, target)
        invalid_ids = ids.clone()
        invalid_ids[0, -1] = 4
        with self.assertRaises(ValueError):
            scorer(invalid_ids, target, torch.zeros_like(target))
        wrong_mask = target.clone()
        wrong_mask[0, -1] = False
        with self.assertRaises(ValueError):
            scorer(ids, wrong_mask, torch.zeros_like(target))
        with self.assertRaises(ValueError):
            scorer(ids.int(), target, torch.zeros_like(target))

    def test_advantages_group_centered_without_standard_deviation(self):
        rewards = torch.tensor([1., 0., 1., 0., 1., 1., 1., 0.])
        expected = torch.tensor([.5, -.5, .5, -.5, .25, .25, .25, -.75])
        self.assertTrue(torch.equal(group_advantages(rewards, 4), expected))

    def test_uniform_groups_have_zero_advantage(self):
        rewards = torch.tensor([0., 0., 0., 0., 1., 1., 1., 1.])
        self.assertTrue(torch.equal(group_advantages(rewards, 4), torch.zeros(8)))

    def test_group_advantages_require_complete_finite_groups(self):
        for rewards, size in ((torch.zeros(3), 2), (torch.zeros(4), 1),
                              (torch.zeros((2, 2)), 2), (torch.ones(4, dtype=torch.long), 2)):
            with self.subTest(shape=rewards.shape, size=size), self.assertRaises(ValueError):
                group_advantages(rewards, size)
        with self.assertRaises(FloatingPointError):
            group_advantages(torch.tensor([0., float("nan")]), 2)

    def test_policy_gradient_direction_positive_and_negative_rewards(self):
        current = torch.tensor([[-2., -3.], [-2., -3.]], requires_grad=True)
        result = self.loss(current, current.detach().clone(), torch.tensor([0.5, -0.5]))
        result["loss_sum"].backward()
        self.assertTrue(bool((current.grad[0] < 0).all()))
        self.assertTrue(bool((current.grad[1] > 0).all()))

    def test_clipping_flattens_only_advantage_improving_direction(self):
        # Positive A: ratios above 1.2 clip. Negative A: ratios below .8 clip.
        old = torch.full((4, 1), -3.)
        current = (old + torch.tensor([[math.log(1.5)], [math.log(.5)],
                                       [math.log(.5)], [math.log(1.5)]])).requires_grad_()
        result = self.loss(current, old, torch.tensor([1., -1., 1., -1.]))
        result["loss_sum"].backward()
        self.assertEqual(float(current.grad[0]), 0)
        self.assertEqual(float(current.grad[1]), 0)
        self.assertLess(float(current.grad[2]), 0)
        self.assertGreater(float(current.grad[3]), 0)
        self.assertEqual(int(result["clip_count"]), 2)
        self.assertEqual(int(result["outside_clip_count"]), 4)

    def test_identical_policies_have_zero_kl_and_unit_ratios(self):
        current = torch.tensor([[-2., -3.]], requires_grad=True)
        result = self.loss(current, current.detach(), torch.zeros(1), beta=.04)
        self.assertEqual(float(result["loss_sum"]), 0)
        self.assertEqual(float(result["kl_sum"]), 0)
        self.assertEqual(float(result["ratio_sum"]), 2)
        result["loss_sum"].backward()
        self.assertTrue(torch.equal(current.grad, torch.zeros_like(current)))

    def test_kl_pulls_current_back_toward_reference_with_zero_advantage(self):
        current = torch.tensor([[-2., -4.]], requires_grad=True)
        reference = torch.tensor([[-3., -3.]])
        result = self.loss(current, current.detach(), torch.zeros(1), ref=reference, beta=.04)
        result["loss_sum"].backward()
        self.assertGreater(float(current.grad[0, 0]), 0)
        self.assertLess(float(current.grad[0, 1]), 0)
        self.assertGreater(float(result["kl_sum"]), 0)

    def test_prompt_values_do_not_enter_loss_or_gradient_even_when_nonfinite(self):
        current = torch.tensor([[float("nan"), -2., -3.]], requires_grad=True)
        old = torch.tensor([[float("inf"), -2., -3.]])
        mask = torch.tensor([[False, True, True]])
        result = self.loss(current, old, torch.ones(1), mask, beta=.04)
        self.assertTrue(bool(torch.isfinite(result["loss_sum"])))
        self.assertEqual(int(result["target_tokens"]), 2)
        result["loss_sum"].backward()
        self.assertEqual(float(current.grad[0, 0]), 0)

    def test_old_reference_and_advantage_are_constants(self):
        current = torch.full((1, 2), -2., requires_grad=True)
        old = torch.full((1, 2), -2., requires_grad=True)
        reference = torch.full((1, 2), -2.1, requires_grad=True)
        advantage = torch.ones(1, requires_grad=True)
        result = self.loss(current, old, advantage, ref=reference, beta=.04)
        result["loss_sum"].backward()
        self.assertIsNone(old.grad)
        self.assertIsNone(reference.grad)
        self.assertIsNone(advantage.grad)
        for name, value in result.items():
            if name != "loss_sum":
                self.assertFalse(value.requires_grad)

    def test_global_sum_equivalence_for_unequal_microbatch_token_counts(self):
        mask = torch.tensor([[False, True, True], [False, False, True]])
        old = torch.full((2, 3), -2.)
        current = (old + .1).requires_grad_()
        advantage = torch.tensor([.5, -.5])
        full = self.loss(current, old, advantage, mask, beta=.04)
        parts = [self.loss(current[i:i+1], old[i:i+1], advantage[i:i+1],
                           mask[i:i+1], beta=.04) for i in range(2)]
        self.assertTrue(torch.allclose(full["loss_sum"], sum(p["loss_sum"] for p in parts)))
        self.assertEqual(int(full["target_tokens"]), sum(int(p["target_tokens"]) for p in parts))

    def test_nonfinite_targets_and_overflow_fail_without_silent_clamping(self):
        with self.assertRaises(FloatingPointError):
            self.loss(torch.tensor([[float("nan")]]), torch.zeros((1, 1)), torch.ones(1))
        with self.assertRaises(FloatingPointError):
            self.loss(torch.zeros((1, 1)), torch.full((1, 1), -1000.), torch.ones(1))
        with self.assertRaises(FloatingPointError):
            self.loss(torch.full((1, 1), -1000.), torch.full((1, 1), -1000.),
                      torch.ones(1), ref=torch.zeros((1, 1)), beta=.04)

    def test_shape_dtype_and_coefficient_errors_are_explicit(self):
        current = torch.full((1, 2), -2.)
        mask = torch.ones_like(current, dtype=torch.bool)
        with self.assertRaises(ValueError):
            self.loss(current, current[:, :1], torch.ones(1))
        with self.assertRaises(ValueError):
            self.loss(current, current, torch.ones(2))
        for epsilon in (0, 1, True, -.1, float("nan")):
            with self.subTest(epsilon=epsilon), self.assertRaises(ValueError):
                policy_loss(current, current, current, torch.ones(1), mask, epsilon, .04)
        with self.assertRaises(ValueError):
            policy_loss(current, current, current, torch.ones(1), mask, .2, -.1)


if __name__ == "__main__":
    unittest.main()
