"""Regression tests for success history and asynchronous vector-environment resets."""
from pathlib import Path
import sys
import unittest
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fr5_pick_place.task_logic import (EpisodeProgress, completed_placement_candidate,
                                       controlled_placement_candidate, placement_candidate,
                                       staged_rewards)


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

    def test_thrown_object_cannot_latch_controlled_placement(self):
        goal = torch.tensor([[0.0, 0.0, 0.115]]).repeat(4, 1)
        pos = goal.clone()
        velocity = torch.zeros(4, 3)
        velocity[1, 2] = -0.25
        fingers = torch.full((4, 2), 0.025)
        fingers[2] = 0.0
        distance = torch.full((4,), 0.03)
        distance[3] = 0.08
        actual = controlled_placement_candidate(
            torch.ones(4, dtype=torch.bool), pos, goal, velocity, fingers, distance)
        self.assertEqual(actual.tolist(), [True, False, False, False])

    def test_success_requires_controlled_place_and_release_history(self):
        pos = torch.zeros(3, 3)
        goal = torch.zeros_like(pos)
        vel = torch.zeros_like(pos)
        fingers = torch.zeros(3, 2)
        distance = torch.full((3,), 0.10)
        actual = completed_placement_candidate(
            torch.tensor([False, True, True]), torch.tensor([False, False, True]),
            torch.ones(3, dtype=torch.bool), pos, goal, vel, fingers, distance)
        self.assertEqual(actual.tolist(), [False, False, True])

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

    def test_stationary_late_stages_pay_no_positive_reward(self):
        goal = torch.tensor([[0.0, 0.0, 0.115]])
        pos = torch.tensor([[0.0, 0.02, 0.160]])
        rewards = staged_rewards(pos, goal, torch.full((1, 2), 0.025),
                                 torch.tensor([0.02]), torch.tensor([True]), 0.115)
        self.assertEqual(rewards["transport"].item(), 0.0)
        self.assertEqual(rewards["lower"].item(), 0.0)
        self.assertEqual(rewards["withdraw"].item(), 0.0)

    def test_progress_reward_is_control_frequency_independent(self):
        goal = torch.tensor([[0.0, 0.0, 0.115]])
        common = dict(object_pos=goal, goal_pos=goal,
                      fingers=torch.full((1, 2), 0.025),
                      tcp_distance=torch.tensor([0.02]), lifted=torch.tensor([True]),
                      rest_height=0.115, lower_z_gain=torch.tensor([0.10]))
        at_50_hz = staged_rewards(step_dt=0.02, **common)["lower"] * 0.02
        at_100_hz = staged_rewards(step_dt=0.01, **common)["lower"] * 0.01
        self.assertTrue(torch.allclose(at_50_hz, at_100_hz))

    def test_xy_regression_does_not_cancel_new_vertical_progress(self):
        memory = EpisodeProgress(1, "cpu")
        false = torch.tensor([False])
        true = torch.tensor([True])
        candidate = lambda lifted: false
        memory.update(1, 0.02, false, candidate,
                      lower_xy_score=torch.tensor([1.0]), lower_xy_active=true,
                      lower_z_score=torch.tensor([0.2]), lower_z_active=true)
        memory.update(2, 0.02, false, candidate,
                      lower_xy_score=torch.tensor([0.8]), lower_xy_active=true,
                      lower_z_score=torch.tensor([0.3]), lower_z_active=true)
        self.assertEqual(memory.lower_xy_gain.item(), 0.0)
        self.assertAlmostEqual(memory.lower_z_gain.item(), 0.1, places=6)

    def test_release_reward_requires_lift_and_placement(self):
        pos = torch.tensor([[0.0, 0.0, 0.115]]).repeat(3, 1)
        goal = pos.clone()
        fingers = torch.full((3, 2), 0.020)
        distance = torch.full((3,), 0.03)
        pos[2, 1] = 0.03
        lifted = torch.tensor([False, True, True])
        placed_once = torch.tensor([False, True, False])
        rewards = staged_rewards(pos, goal, fingers, distance, lifted, 0.115,
                                 placed_once=placed_once, just_placed=placed_once)
        self.assertEqual(rewards["release"][[0, 2]].tolist(), [0.0, 0.0])
        self.assertAlmostEqual(rewards["release"][1].item(), 149.9, places=4)
        self.assertEqual(rewards["lower"][:2].tolist(), [0.0, 0.0])
        self.assertEqual(rewards["lower"][2].item(), 0.0)

    def test_opening_is_rewarded_only_after_reaching_placement(self):
        goal = torch.tensor([[0.0, 0.0, 0.115]]).repeat(3, 1)
        pos = goal.clone()
        fingers = torch.tensor([[0.020, 0.020], [0.008, 0.008], [0.0, 0.0]])
        distance = torch.full((3,), 0.03)
        rewards = staged_rewards(
            pos, goal, fingers, distance, torch.tensor([True, True, True]), 0.115,
            placed_once=torch.tensor([True, True, True]),
            released_once=torch.tensor([False, False, True]),
            just_released=torch.tensor([False, False, True]),
            opening_gain=torch.tensor([0.0, 0.33333334, 0.0]),
        )
        self.assertLess(rewards["release"][0].item(), 0.0)
        self.assertGreater(rewards["release"][1].item(), 0.0)
        self.assertEqual(rewards["release"][2].item(), 250.0)
        self.assertEqual(rewards["withdraw"][2].item(), 0.0)

    def test_repeated_open_state_cannot_be_farmed(self):
        goal = torch.tensor([[0.0, 0.0, 0.115]])
        pos = goal.clone()
        rewards = staged_rewards(pos, goal, torch.full((1, 2), 0.020),
                                 torch.tensor([0.03]), torch.tensor([True]), 0.115,
                                 placed_once=torch.tensor([True]), opening_gain=torch.zeros(1))
        self.assertAlmostEqual(rewards["release"].item(), -0.10, places=6)

    def test_new_best_progress_cannot_be_replayed(self):
        memory = EpisodeProgress(1, "cpu")
        inactive = torch.tensor([False])
        active = torch.tensor([True])
        score = torch.tensor([0.4])
        memory.update(1, 0.02, inactive, lambda lifted: inactive,
                      transport_score=score, transport_active=active)
        self.assertAlmostEqual(memory.transport_gain.item(), 0.4)
        memory.update(2, 0.02, inactive, lambda lifted: inactive,
                      transport_score=score, transport_active=active)
        self.assertEqual(memory.transport_gain.item(), 0.0)
        memory.update(3, 0.02, inactive, lambda lifted: inactive,
                      transport_score=torch.tensor([0.2]), transport_active=active)
        self.assertEqual(memory.transport_gain.item(), 0.0)
        memory.update(4, 0.02, inactive, lambda lifted: inactive,
                      transport_score=torch.tensor([0.5]), transport_active=active)
        self.assertAlmostEqual(memory.transport_gain.item(), 0.1, places=6)

    def test_grasp_and_lift_milestones_pay_once(self):
        memory = EpisodeProgress(1, "cpu")
        false = torch.tensor([False])
        true = torch.tensor([True])
        candidate = lambda lifted: false
        memory.update(1, 0.02, false, candidate, grasping_now=true,
                      reach_score=torch.tensor([0.8]), reach_active=true)
        self.assertTrue(memory.just_grasped.item())
        self.assertAlmostEqual(memory.reach_gain.item(), 0.8)
        memory.update(2, 0.02, false, candidate, grasping_now=true,
                      reach_score=torch.tensor([0.8]), reach_active=true)
        self.assertFalse(memory.just_grasped.item())
        self.assertEqual(memory.reach_gain.item(), 0.0)
        memory.update(3, 0.02, true, candidate, grasping_now=true,
                      lift_score=torch.tensor([1.0]), lift_active=true)
        self.assertTrue(memory.just_lifted.item())
        memory.update(4, 0.02, true, candidate, grasping_now=true,
                      lift_score=torch.tensor([1.0]), lift_active=true)
        self.assertFalse(memory.just_lifted.item())

    def test_placement_and_release_are_one_time_milestones(self):
        memory = EpisodeProgress(1, "cpu")
        candidate = lambda lifted: torch.tensor([False])
        opening = torch.tensor([0.0])
        memory.update(1, 0.02, torch.tensor([True]), candidate,
                      placement_now=torch.tensor([True]), release_now=torch.tensor([False]), opening=opening)
        self.assertTrue(memory.just_placed.item())
        memory.update(2, 0.02, torch.tensor([False]), candidate,
                      placement_now=torch.tensor([True]), release_now=torch.tensor([False]), opening=opening)
        self.assertFalse(memory.just_placed.item())
        memory.update(3, 0.02, torch.tensor([False]), candidate,
                      placement_now=torch.tensor([True]), release_now=torch.tensor([True]),
                      opening=torch.tensor([1.0]))
        self.assertTrue(memory.just_released.item())
        memory.update(4, 0.02, torch.tensor([False]), candidate,
                      placement_now=torch.tensor([True]), release_now=torch.tensor([True]),
                      opening=torch.tensor([1.0]))
        self.assertFalse(memory.just_released.item())


if __name__ == "__main__":
    unittest.main()
