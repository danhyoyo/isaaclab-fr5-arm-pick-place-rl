# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.utils.math import combine_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_is_lifted(
    env: ManagerBasedRLEnv, minimal_height: float, object_cfg: SceneEntityCfg = SceneEntityCfg("object")
) -> torch.Tensor:
    """Reward the agent for lifting the object above the minimal height."""
    object: RigidObject = env.scene[object_cfg.name]
    
    return torch.where(object.data.root_pos_w.torch[:, 2] > minimal_height, 1.0, 0.0)


def object_ee_distance(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward the agent for reaching the object using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    # Target object position: (num_envs, 3)
    cube_pos_w = object.data.root_pos_w.torch
    # End-effector position: (num_envs, 3)
    ee_w = ee_frame.data.target_pos_w.torch[..., 0, :]
    # Distance of the end-effector to the object: (num_envs,)
    object_ee_distance = torch.norm(cube_pos_w - ee_w, dim=1)

    return 1 - torch.tanh(object_ee_distance / std)


def object_is_lifted_dist_limit(
    env: ManagerBasedRLEnv,
    minimal_height: float,
    minimal_dist: float,
    target_region_cfg: SceneEntityCfg = SceneEntityCfg("target_region"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object")
) -> torch.Tensor:
    """Reward the agent for lifting the object above the minimal height."""
    target_region: RigidObject = env.scene[target_region_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    distance = torch.norm(target_region.data.root_pos_w.torch - object.data.root_pos_w.torch, dim=1)
    
    condition = (
        (distance >= minimal_dist) & (object.data.root_pos_w.torch[:, 2] > minimal_height)
    )

    height_reward = torch.clamp(
        (object.data.root_pos_w.torch[:, 2] - minimal_height) / 0.05,
        min=0.0,
        max=1.0
    )

    return condition.float() * height_reward


def object_goal_distance(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward the agent for tracking the goal pose using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    # compute the desired position in the world frame
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(robot.data.root_pos_w.torch, robot.data.root_quat_w.torch, des_pos_b)
    # distance of the end-effector to the object: (num_envs,)
    distance = torch.norm(des_pos_w - object.data.root_pos_w.torch, dim=1)
    # rewarded if the object is lifted above the threshold
    return (object.data.root_pos_w.torch[:, 2] > minimal_height) * (1 - torch.tanh(distance / std))


def object_target_region_distance(
    env: ManagerBasedRLEnv,
    std: float,
    target_region_cfg: SceneEntityCfg = SceneEntityCfg("target_region"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward the agent for tracking the goal pose using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    target_region: RigidObject = env.scene[target_region_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    # distance of the end-effector to the object: (num_envs,)
    distance = torch.norm(target_region.data.root_pos_w.torch - object.data.root_pos_w.torch, dim=1)
    # rewarded if the object is lifted above the threshold
    return 1 - torch.tanh(distance / std)


def object_lowering(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_dist: float,
    target_region_cfg: SceneEntityCfg = SceneEntityCfg("target_region"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:

    target_region: RigidObject = env.scene[target_region_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]

    # Current cube height
    current_height = object.data.root_pos_w.torch[:, 2]

    # Initialize buffer if needed
    if not hasattr(env, "prev_cube_height"):
        env.prev_cube_height = current_height.clone()

    # Previous height
    prev_height = env.prev_cube_height

    # Positive when cube goes downward
    height_delta = prev_height - current_height

    # Distance to target
    distance = torch.norm(
        target_region.data.root_pos_w.torch - object.data.root_pos_w.torch,
        dim=1
    )

    # Reward only when:
    # 1. cube is near target
    # 2. cube moved downward
    reward = (
        1000 * (distance < minimal_dist) * torch.clamp(height_delta, min=0.0) * (1 - torch.tanh(distance / std))
    )

    # Update previous height
    env.prev_cube_height = current_height.clone()

    return reward


def object_placed(
    env: ManagerBasedRLEnv,
    target_region_cfg: SceneEntityCfg = SceneEntityCfg("target_region"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    
    target_region: RigidObject = env.scene[target_region_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]

    xy_dist = torch.norm(
        target_region.data.root_pos_w.torch[:, :2]
        - object.data.root_pos_w.torch[:, :2],
        dim=1
    )

    z = object.data.root_pos_w.torch[:, 2]

    success = (
        (xy_dist < 0.025)
        & (z < 0.03)
    )

    return success.float()