"""Regression tests for success history and asynchronous vector-environment resets."""
from pathlib import Path
import sys
import unittest
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fr5_pick_place.task_logic import EpisodeProgress, placement_candidate


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


if __name__ == "__main__":
    unittest.main()
