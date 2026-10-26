"""FR5 observations, staged rewards, reset sampling and terminal conditions."""
import torch
from isaaclab.envs.mdp import *  # Standard action-rate, observation, and timeout helpers.
from isaaclab.utils.math import quat_apply
from .constants import (TCP_OFFSET, REST_HEIGHT, GOAL_MARKER_Z, LIFT_HEIGHT, RELEASE_XY_TOL,
                        RELEASE_Z_TOL, LOWER_XY_ENTRY_TOL, CONTROLLED_LOWER_MAX_SPEED,
                        CONTROLLED_PLACE_MAX_SPEED, SUCCESS_HOLD_SECONDS, PICK_CENTER,
                        GOAL_CENTER, TABLE_TOP)
from .task_logic import (EpisodeProgress, completed_placement_candidate,
                         controlled_placement_candidate, staged_rewards)


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
    closed = fingers.mean(-1) > 0.012
    grasping = (distance < 0.025) & closed
    lifting = ((pos[:, 2] > LIFT_HEIGHT) & (distance < 0.045) & closed)
    lifted_now = memory.lifted | lifting
    xy_goal = torch.linalg.vector_norm(pos[:, :2] - goal[:, :2], dim=-1)
    object_speed = torch.linalg.vector_norm(velocity, dim=-1)
    release_zone_now = (lifted_now & (xy_goal < RELEASE_XY_TOL)
                        & ((pos[:, 2] - goal[:, 2]).abs() < RELEASE_Z_TOL))
    # Placement is latched only while the gripper still controls a slow object.
    # A thrown object merely passing through the release zone cannot earn it.
    placement_now = controlled_placement_candidate(
        lifted_now, pos, goal, velocity, fingers, distance,
        RELEASE_XY_TOL, RELEASE_Z_TOL, CONTROLLED_PLACE_MAX_SPEED)
    open_now = fingers.abs().amax(dim=-1) < 0.004
    opening = ((0.012 - fingers.abs().mean(-1)) / 0.012).clamp(0, 1)
    controlled_hold = closed & (distance < 0.045)
    held = controlled_hold & (pos[:, 2] > REST_HEIGHT + 0.020)
    near_goal = xy_goal < LOWER_XY_ENTRY_TOL
    goal_progress = 1 - torch.tanh(xy_goal / 0.12)
    reach_progress = 1 - torch.tanh(distance / 0.10)
    height_progress = ((pos[:, 2] - REST_HEIGHT) / 0.045).clamp(0, 1)
    # First refine XY from the entry radius to the release radius.  Only after
    # XY is inside the release radius does vertical descent earn progress.
    lower_xy_progress = ((LOWER_XY_ENTRY_TOL - xy_goal)
                         / (LOWER_XY_ENTRY_TOL - RELEASE_XY_TOL)).clamp(0, 1)
    lower_z_error = (pos[:, 2] - REST_HEIGHT).abs()
    placed_now_or_before = memory.placed | placement_now
    release_now = (placed_now_or_before & release_zone_now & open_now
                   & (object_speed < CONTROLLED_PLACE_MAX_SPEED))
    released_now = memory.released | release_now
    memory.update(env.common_step_counter, env.step_dt, lifting,
                  lambda lifted: completed_placement_candidate(
                      memory.placed, memory.released, lifted, pos, goal, velocity,
                      fingers, distance),
                   grasping_now=grasping, placement_now=placement_now,
                   release_now=release_now, opening=opening,
                   reach_score=reach_progress, reach_active=~memory.grasped,
                   lift_score=height_progress,
                   lift_active=(memory.grasped | grasping) & closed & (distance < 0.045) & ~memory.lifted,
                   transport_score=goal_progress,
                   transport_active=lifted_now & held & ~near_goal,
                   lower_xy_score=lower_xy_progress,
                   lower_xy_active=(lifted_now & near_goal & ~memory.placed & controlled_hold
                                    & (object_speed < CONTROLLED_LOWER_MAX_SPEED)),
                   lower_z_error=lower_z_error,
                   lower_z_active=(lifted_now & (xy_goal < RELEASE_XY_TOL) & ~memory.placed
                                   & controlled_hold
                                   & (object_speed < CONTROLLED_LOWER_MAX_SPEED)),
                   lower_z_target_error=RELEASE_Z_TOL,
                   opening_active=(placed_now_or_before & release_zone_now & ~memory.released
                                   & (object_speed < CONTROLLED_PLACE_MAX_SPEED)),
                   withdraw_score=(distance / 0.10).clamp(0, 1),
                   withdraw_active=(released_now & release_zone_now & open_now
                                    & (object_speed < CONTROLLED_PLACE_MAX_SPEED)))
    env.fr5_state_cache = (tcp, quat, pos, goal, fingers, velocity, distance, memory)
    env.fr5_state_stamp = stamp
    return env.fr5_state_cache


def task_observation(env):
    tcp, quat, pos, goal, fingers, velocity, distance, memory = state(env)
    arm = env.action_manager.get_term("arm")
    current_b, _ = arm._compute_frame_pose()
    command_error = arm.target_position - current_b
    return torch.cat([tcp, quat, pos, goal, pos - tcp, goal - pos, velocity, command_error,
                      memory.lifted[:, None].float(), memory.placed[:, None].float()], -1)


def task_reward(env, stage: str):
    tcp, quat, pos, goal, fingers, velocity, distance, memory = state(env)
    if stage == "success":
        return (memory.hold >= SUCCESS_HOLD_SECONDS).float()
    if stage == "time":
        return torch.ones(env.num_envs, device=env.device)
    rewards = staged_rewards(pos, goal, fingers, distance, memory.lifted, REST_HEIGHT,
                             memory.placed, memory.released, memory.just_grasped,
                             memory.just_lifted, memory.just_placed, memory.just_released,
                             reach_gain=memory.reach_gain, lift_gain=memory.lift_gain,
                             transport_gain=memory.transport_gain,
                             lower_xy_gain=memory.lower_xy_gain,
                             lower_z_gain=memory.lower_z_gain,
                             opening_gain=memory.opening_gain,
                             withdraw_gain=memory.withdraw_gain, step_dt=env.step_dt,
                             release_xy_tol=RELEASE_XY_TOL, release_z_tol=RELEASE_Z_TOL)
    try:
        return rewards[stage]
    except KeyError as exc:
        raise ValueError(stage) from exc


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
