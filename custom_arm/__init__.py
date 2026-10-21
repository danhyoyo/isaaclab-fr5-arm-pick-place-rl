# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import gymnasium as gym

from . import agents

##
# Joint Position Control Lift Place
##

gym.register(
    id="Isaac-Lift-Cube-Place-Custom-Arm-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_place_env_cfg:CustomArmCubeLiftPlaceEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LiftCubePlacePPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Lift-Cube-Place-Custom-Arm-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_place_env_cfg:CustomArmCubeLiftPlaceEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LiftCubePlacePPORunnerCfg",
    },
    disable_env_checker=True,
)
