"""Simulator-independent success logic and per-environment episode memory."""
import torch


def placement_candidate(lifted, object_pos, goal_pos, object_velocity, fingers, tcp_distance):
    xy = torch.linalg.vector_norm(object_pos[:, :2] - goal_pos[:, :2], dim=-1)
    return (lifted & (xy < 0.020) & ((object_pos[:, 2] - goal_pos[:, 2]).abs() < 0.008)
            & (torch.linalg.vector_norm(object_velocity, dim=-1) < 0.06)
            & (fingers.abs().amax(dim=-1) < 0.004) & (tcp_distance > 0.055))


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
