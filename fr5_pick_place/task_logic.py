"""Simulator-independent success logic and per-environment episode memory."""
import torch


def placement_candidate(lifted, object_pos, goal_pos, object_velocity, fingers, tcp_distance):
    xy = torch.linalg.vector_norm(object_pos[:, :2] - goal_pos[:, :2], dim=-1)
    return (lifted & (xy < 0.020) & ((object_pos[:, 2] - goal_pos[:, 2]).abs() < 0.008)
            & (torch.linalg.vector_norm(object_velocity, dim=-1) < 0.06)
            & (fingers.abs().amax(dim=-1) < 0.004) & (tcp_distance > 0.055))


def staged_rewards(object_pos, goal_pos, fingers, tcp_distance, lifted, rest_height):
    """Return mutually exclusive shaping rewards for the pick-place stages.

    Each completed stage stops paying reward.  The next stage has a larger base
    reward, so the policy cannot earn more by hovering just before a transition.
    """
    xy_goal = torch.linalg.vector_norm(object_pos[:, :2] - goal_pos[:, :2], dim=-1)
    z_error = (object_pos[:, 2] - rest_height).abs()
    closed = fingers.mean(-1) > 0.012
    open_gripper = (fingers.abs().amax(dim=-1) < 0.004)
    held = closed & (tcp_distance < 0.045) & (object_pos[:, 2] > rest_height + 0.020)
    near_goal = xy_goal < 0.035
    placed = lifted & (xy_goal < 0.020) & (z_error < 0.008)

    before_lift = ~lifted
    height_progress = ((object_pos[:, 2] - rest_height) / 0.045).clamp(0, 1)
    goal_progress = 1 - torch.tanh(xy_goal / 0.12)
    xy_precision = 1 - torch.tanh(xy_goal / 0.035)
    lower_progress = 1 - torch.tanh(z_error / 0.05)
    open_progress = (1 - fingers.abs().mean(-1) / 0.04).clamp(0, 1)

    return {
        "reach": before_lift.float() * (1 - torch.tanh(tcp_distance / 0.10)),
        "grasp": before_lift.float() * (tcp_distance < 0.025).float() * closed.float(),
        "lift": before_lift.float() * closed.float() * (tcp_distance < 0.045).float() * height_progress,
        # The base rewards make crossing each stage boundary preferable to farming
        # the preceding dense reward indefinitely.
        "transport": lifted.float() * held.float() * (~near_goal).float() * (1 + goal_progress),
        "lower": lifted.float() * near_goal.float() * (2 + xy_precision + lower_progress),
        "release": placed.float() * open_progress,
        "withdraw": placed.float() * open_gripper.float() * (tcp_distance / 0.10).clamp(0, 1),
    }


class EpisodeProgress:
    def __init__(self, num_envs, device):
        self.lifted = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.hold = torch.zeros(num_envs, device=device)
        self.last_step = torch.full((num_envs,), -1, dtype=torch.long, device=device)

    def reset(self, ids, step):
        self.lifted[ids] = False
        self.hold[ids] = 0
        self.last_step[ids] = step

    def update(self, step, dt, lifting_now, candidate_fn):
        fresh = self.last_step != step
        self.lifted |= lifting_now & fresh
        candidate = candidate_fn(self.lifted)
        self.hold[fresh] = torch.where(candidate[fresh], self.hold[fresh] + dt, 0.0)
        self.last_step[fresh] = step
