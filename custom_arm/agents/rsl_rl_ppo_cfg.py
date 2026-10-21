# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlMLPModelCfg, RslRlPpoAlgorithmCfg


@configclass
class LiftCubePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    obs_groups = {"actor": ["policy"], "critic": ["policy"]}
    num_steps_per_env = 24
    max_iterations = 1500
    save_interval = 50
    experiment_name = "franka_lift"
    actor = RslRlMLPModelCfg(
        hidden_dims=[256, 128, 64],
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0),
        activation="elu",
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[256, 128, 64],
        obs_normalization=False,
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.006,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.98,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )

@configclass
class LiftCubePlacePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    obs_groups = {"actor": ["policy"], "critic": ["policy"]}
    num_steps_per_env = 24
    max_iterations = 1500
    save_interval = 50
    experiment_name = "franka_lift_place_custom_arm"
    actor = RslRlMLPModelCfg(
        hidden_dims=[128, 64, 32],
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0),
        activation="elu",
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[128, 64, 32],
        obs_normalization=True,
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.006,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=8.0e-5,
        schedule="adaptive",
        gamma=0.98,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=0.4,
    )
