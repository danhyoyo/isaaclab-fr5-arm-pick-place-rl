"""Regression tests for success history and asynchronous vector-environment resets."""
from pathlib import Path
import sys
import unittest
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fr5_pick_place.task_logic import EpisodeProgress, placement_candidate, staged_rewards


class TaskLogicTest(unittest.TestCase):
    def test_push_without_lift_cannot_succeed(self):
        lifted = torch.tensor([False, True, True, True, True])
        pos = torch.zeros(5, 3)
        goal = torch.zeros_like(pos)
        vel = torch.zeros_like(pos)
        vel[2, 0] = 0.2
        fingers = torch.zeros(5, 2)
        fingers[3] = 0.025
        distance = torch.tensor([0.1, 0.1, 0.1, 0.1, 0.02])
        actual = placement_candidate(lifted, pos, goal, vel, fingers, distance)
        self.assertEqual(actual.tolist(), [False, True, False, False, False])

    def test_history_updates_once_per_step_and_partial_reset(self):
        memory = EpisodeProgress(2, "cpu")
        candidate = lambda lifted: lifted
        memory.update(1, 0.02, torch.tensor([True, True]), candidate)
        memory.update(1, 0.02, torch.tensor([True, True]), candidate)
        self.assertTrue(torch.allclose(memory.hold, torch.tensor([0.02, 0.02])))
        memory.reset(torch.tensor([0]), 1)
        memory.update(1, 0.02, torch.tensor([False, False]), candidate)
        self.assertEqual(memory.lifted.tolist(), [False, True])
        self.assertTrue(torch.allclose(memory.hold, torch.tensor([0.0, 0.02])))
        memory.update(2, 0.02, torch.tensor([False, False]), candidate)
        self.assertTrue(torch.allclose(memory.hold, torch.tensor([0.0, 0.04])))

    def test_hold_must_be_continuous(self):
        memory = EpisodeProgress(1, "cpu")
        memory.update(1, 0.02, torch.tensor([True]), lambda lifted: lifted)
        memory.update(2, 0.02, torch.tensor([False]), lambda lifted: torch.tensor([False]))
        self.assertEqual(memory.hold.item(), 0.0)

    def test_completed_lift_cannot_be_farmed_by_raising_higher(self):
        goal = torch.tensor([[0.0, 0.24, 0.115], [0.0, 0.24, 0.115]])
        pos = torch.tensor([[0.0, 0.0, 0.160], [0.0, 0.0, 0.500]])
        fingers = torch.full((2, 2), 0.025)
        distance = torch.full((2,), 0.02)
        rewards = staged_rewards(pos, goal, fingers, distance, torch.tensor([True, True]), 0.115)
        self.assertEqual(rewards["lift"].tolist(), [0.0, 0.0])
        self.assertTrue(torch.allclose(rewards["transport"][0], rewards["transport"][1]))

    def test_next_stage_pays_more_than_hovering_before_boundary(self):
        fingers = torch.full((2, 2), 0.025)
        distance = torch.full((2,), 0.02)
        lifted = torch.tensor([True, True])
        # First sample hovers just outside the goal region; second enters it at
        # the same height. Entering the lower stage must be more attractive.
        goal = torch.tensor([[0.0, 0.0, 0.115], [0.0, 0.0, 0.115]])
        pos = torch.tensor([[0.0, 0.0351, 0.160], [0.0, 0.0349, 0.160]])
        rewards = staged_rewards(pos, goal, fingers, distance, lifted, 0.115)
        self.assertGreater(rewards["lower"][1].item(), rewards["transport"][0].item())

    def test_release_reward_requires_lift_and_placement(self):
        pos = torch.tensor([[0.0, 0.0, 0.115]]).repeat(3, 1)
        goal = pos.clone()
        fingers = torch.zeros(3, 2)
        distance = torch.full((3,), 0.07)
        pos[2, 1] = 0.03
        rewards = staged_rewards(pos, goal, fingers, distance,
                                 torch.tensor([False, True, True]), 0.115)
        self.assertEqual(rewards["release"].tolist(), [0.0, 1.0, 0.0])


if __name__ == "__main__":
    unittest.main()
