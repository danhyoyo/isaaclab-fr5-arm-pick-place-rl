# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject, Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """The position of the object in the robot's root frame."""
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    object_pos_w = object.data.root_pos_w.torch[:, :3]
    object_pos_b, _ = subtract_frame_transforms(robot.data.root_pos_w.torch, robot.data.root_quat_w.torch, object_pos_w)
    return object_pos_b


def target_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    target_region_cfg: SceneEntityCfg = SceneEntityCfg("target_region"),
) -> torch.Tensor:
    """The position of the object in the robot's root frame."""
    robot: RigidObject = env.scene[robot_cfg.name]
    target_region: RigidObject = env.scene[target_region_cfg.name]
    object_pos_w = target_region.data.root_pos_w.torch[:, :3]
    object_pos_b, _ = subtract_frame_transforms(robot.data.root_pos_w.torch, robot.data.root_quat_w.torch, object_pos_w)
    return object_pos_b


def gray_img(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Outputs stacked grayscale images (current + previous)."""

    asset: Articulation = env.scene[asset_cfg.name]

    # RGB tensor from camera
    rgb = asset.data.output["rgb"]  # shape: (num_envs, H, W, 3)

    # move channel to PyTorch format
    rgb = rgb.permute(0, 3, 1, 2).float() / 255.0  # (N,3,H,W)

    # grayscale conversion (vectorized)
    gray = (
        0.299 * rgb[:, 0:1] +
        0.587 * rgb[:, 1:2] +
        0.114 * rgb[:, 2:3]
    )  # (N,1,H,W)

    return gray