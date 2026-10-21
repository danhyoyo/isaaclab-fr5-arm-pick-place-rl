"""Standalone FR5 pick-and-place task registrations (no dependency on pick_and_place)."""
import gymnasium as gym

TASK = "Isaac-FR5-PickPlace-v0"
PLAY_TASK = "Isaac-FR5-PickPlace-Play-v0"
for name, config in [(TASK, "FR5PickPlaceEnvCfg"), (PLAY_TASK, "FR5PickPlacePlayEnvCfg")]:
    if name not in gym.registry:
        gym.register(
            id=name, entry_point="isaaclab.envs:ManagerBasedRLEnv", disable_env_checker=True,
            kwargs={"env_cfg_entry_point": f"{__name__}.env_cfg:{config}",
                    "rsl_rl_cfg_entry_point": f"{__name__}.ppo_cfg:FR5PPORunnerCfg"},
        )
