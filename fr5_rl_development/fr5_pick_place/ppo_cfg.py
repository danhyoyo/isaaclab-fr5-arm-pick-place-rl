"""PPO configuration for four actions: TCP dx/dy/dz plus binary gripper."""
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlMLPModelCfg, RslRlPpoAlgorithmCfg


@configclass
class FR5PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    seed = 42
    obs_groups = {"actor": ["policy"], "critic": ["policy"]}
    num_steps_per_env = 32
    max_iterations = 5000
    save_interval = 100
    experiment_name = "fr5_pick_place"
    clip_actions = 1.0
    actor = RslRlMLPModelCfg(hidden_dims=[256, 128, 64], activation="elu", obs_normalization=True,
              distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.6))
    critic = RslRlMLPModelCfg(hidden_dims=[256, 128, 64], activation="elu", obs_normalization=True)
    algorithm = RslRlPpoAlgorithmCfg(value_loss_coef=1.0, use_clipped_value_loss=True, clip_param=0.2,
                   entropy_coef=0.008, num_learning_epochs=5, num_mini_batches=4, learning_rate=3e-4,
                   schedule="adaptive", gamma=0.99, lam=0.95, desired_kl=0.01, max_grad_norm=1.0)
