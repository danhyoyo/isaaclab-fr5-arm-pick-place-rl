"""Finite integration check of vector origins, full-episode reset and finite observations/rewards."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import json
import torch
import warp as wp
from run import ROOT, register_tasks


def main():
    wp.config.enable_backward = False
    tasks = register_tasks()
    from isaaclab.app import add_launcher_args, launch_simulation
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
    from fr5_pick_place import mdp
    import gymnasium as gym
    parser = argparse.ArgumentParser(description=__doc__)
    add_launcher_args(parser)
    args = parser.parse_args()
    cfg = load_cfg_from_registry(tasks.TASK, "env_cfg_entry_point")
    cfg.scene.num_envs = 4
    cfg.seed = 42
    with launch_simulation(cfg, args):
        env = gym.make(tasks.TASK, cfg=cfg).unwrapped
        try:
            obs, _ = env.reset()
            assert env.action_manager.total_action_dim == 4
            tcp = mdp.geometry(env)[0]
            assert torch.allclose(tcp, tcp[:1].expand_as(tcp), atol=1e-5), tcp
            actions = torch.zeros(env.num_envs, 4, device=env.device)
            actions[:, 3] = 1
            resets = 0
            for step in range(650):
                obs, reward, terminated, truncated, _ = env.step(actions)
                assert torch.isfinite(obs["policy"]).all() and torch.isfinite(reward).all()
                resets += int((terminated | truncated).sum())
                assert not env.termination_manager.get_term("success").any(), "Stationary open gripper cannot succeed"
            assert resets >= env.num_envs, resets
            output = {"passed": True, "steps": 650, "num_envs": env.num_envs,
                      "action_dim": 4, "observation_dim": obs["policy"].shape[-1], "resets": resets}
            (ROOT / "verification").mkdir(exist_ok=True)
            (ROOT / "verification/environment.json").write_text(json.dumps(output, indent=2))
            print(json.dumps(output), flush=True)
        finally:
            env.close()


if __name__ == "__main__":
    main()
