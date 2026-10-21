"""FR5 observations, staged rewards, reset sampling and terminal conditions."""
import torch
from isaaclab.envs.mdp import *  # Standard action-rate, observation, and timeout helpers.
from isaaclab.utils.math import quat_apply
from .constants import (TCP_OFFSET, REST_HEIGHT, GOAL_MARKER_Z, LIFT_HEIGHT, SUCCESS_HOLD_SECONDS,
                        PICK_CENTER, GOAL_CENTER, TABLE_TOP)
from .task_logic import EpisodeProgress, placement_candidate


def progress(env):
    if not hasattr(env, "fr5_progress"):
        env.fr5_progress = EpisodeProgress(env.num_envs, env.device)
        robot = env.scene["robot"]
        env.fr5_hand_id = robot.body_names.index("hand_base_link")
        env.fr5_finger_ids = [robot.joint_names.index(n) for n in ["fj1", "fj2"]]
    return env.fr5_progress


def geometry(env):
    progress(env)
    robot, obj, goal = env.scene["robot"], env.scene["object"], env.scene["target"]
    origins = env.scene.env_origins
    hand = robot.data.body_pos_w.torch[:, env.fr5_hand_id]
    quat = robot.data.body_quat_w.torch[:, env.fr5_hand_id]
    offset = torch.tensor(TCP_OFFSET, device=env.device).expand(env.num_envs, -1)
    tcp = hand + quat_apply(quat, offset) - origins
    pos = obj.data.root_pos_w.torch - origins
    goal_pos = goal.data.root_pos_w.torch - origins
    goal_pos = goal_pos.clone()
    goal_pos[:, 2] += REST_HEIGHT - GOAL_MARKER_Z
    fingers = robot.data.joint_pos.torch[:, env.fr5_finger_ids]
    return tcp, quat, pos, goal_pos, fingers, obj.data.root_lin_vel_w.torch


def state(env):
    stamp = (env.common_step_counter, getattr(env, "fr5_reset_epoch", 0))
    if getattr(env, "fr5_state_stamp", None) == stamp:
        return env.fr5_state_cache
    tcp, quat, pos, goal, fingers, velocity = geometry(env)
    memory = progress(env)
    distance = torch.linalg.vector_norm(tcp - pos, dim=-1)
    lifting = ((pos[:, 2] > LIFT_HEIGHT) & (distance < 0.045) & (fingers.mean(-1) > 0.012))
    memory.update(env.common_step_counter, env.step_dt, lifting,
                  lambda lifted: placement_candidate(lifted, pos, goal, velocity, fingers, distance))
    env.fr5_state_cache = (tcp, quat, pos, goal, fingers, velocity, distance, memory)
    env.fr5_state_stamp = stamp
    return env.fr5_state_cache


def task_observation(env):
    tcp, quat, pos, goal, fingers, velocity, distance, memory = state(env)
    arm = env.action_manager.get_term("arm")
    current_b, _ = arm._compute_frame_pose()
    command_error = arm.target_position - current_b
    return torch.cat([tcp, quat, pos, goal, pos - tcp, goal - pos, velocity, command_error,
                      memory.lifted[:, None].float(), (memory.hold / SUCCESS_HOLD_SECONDS).clamp(0, 1)[:, None]], -1)


def task_reward(env, stage: str):
    tcp, quat, pos, goal, fingers, velocity, distance, memory = state(env)
    xy_goal = torch.linalg.vector_norm(pos[:, :2] - goal[:, :2], dim=-1)
    closed = fingers.mean(-1) > 0.012
    held = closed & (distance < 0.045) & (pos[:, 2] > REST_HEIGHT + 0.020)
    if stage == "reach":
        return 1 - torch.tanh(distance / 0.10)
    if stage == "grasp":
        return (distance < 0.025).float() * closed.float()
    if stage == "lift":
        return held.float() * ((pos[:, 2] - REST_HEIGHT) / 0.08).clamp(0, 1)
    if stage == "transport":
        return memory.lifted.float() * held.float() * (1 - torch.tanh(xy_goal / 0.12))
    if stage == "lower":
        return memory.lifted.float() * (xy_goal < 0.035) * (1 - torch.tanh((pos[:, 2] - REST_HEIGHT).abs() / 0.05))
    if stage == "release":
        placed = memory.lifted & (xy_goal < 0.020) & ((pos[:, 2] - REST_HEIGHT).abs() < 0.008)
        return placed.float() * (1 - fingers.abs().mean(-1) / 0.04).clamp(0, 1)
    if stage == "withdraw":
        placed = memory.lifted & (xy_goal < 0.020) & ((pos[:, 2] - REST_HEIGHT).abs() < 0.008)
        return placed.float() * (fingers.abs().amax(-1) < 0.004) * (distance / 0.10).clamp(0, 1)
    if stage == "success":
        return (memory.hold >= SUCCESS_HOLD_SECONDS).float()
    raise ValueError(stage)


def success(env):
    return state(env)[-1].hold >= SUCCESS_HOLD_SECONDS


def object_out_of_workspace(env):
    pos = env.scene["object"].data.root_pos_w.torch - env.scene.env_origins
    return ((pos[:, 2] < TABLE_TOP - 0.06) | (pos[:, 2] > 0.9)
            | (pos[:, 0] < -0.70) | (pos[:, 0] > -0.20) | (pos[:, 1].abs() > 0.35))


def reset_task(env, env_ids, randomize: bool = True):
    """Reset objects and history only for selected environments, including batched origins."""
    memory = progress(env)
    memory.reset(env_ids, env.common_step_counter)
    env.fr5_reset_epoch = getattr(env, "fr5_reset_epoch", 0) + 1
    count = env.scene.env_origins[env_ids].shape[0]
    for name, center in [("object", PICK_CENTER), ("target", GOAL_CENTER)]:
        asset = env.scene[name]
        pose = asset.data.default_root_pose.torch[env_ids].clone()
        pose[:, :3] = torch.tensor(center, device=env.device)
        if randomize:
            pose[:, 0] += torch.rand(count, device=env.device) * 0.08 - 0.04
            pose[:, 1] += torch.rand(count, device=env.device) * 0.04 - 0.02
        pose[:, :3] += env.scene.env_origins[env_ids]
        asset.write_root_pose_to_sim_index(root_pose=pose, env_ids=env_ids)
        asset.write_root_velocity_to_sim_index(root_velocity=torch.zeros((count, 6), device=env.device), env_ids=env_ids)
