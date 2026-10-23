"""Evaluate a learned FR5 policy over completed episodes and report task success."""
import argparse
import json
from pathlib import Path
import torch
import warp as wp
from run import ROOT, register_tasks


def main():
    wp.config.enable_backward = False
    tasks = register_tasks()
    from isaaclab.app import add_launcher_args, launch_simulation
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
    import gymnasium as gym
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=32)
    parser.add_argument("--num_envs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=123)
    add_launcher_args(parser)
    args = parser.parse_args()
    if args.episodes < 1 or args.num_envs < 1:
        parser.error("episodes and num_envs must be positive")
    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_file():
        parser.error(f"Checkpoint does not exist: {checkpoint}")
    cfg = load_cfg_from_registry(tasks.PLAY_TASK, "env_cfg_entry_point")
    agent_cfg = load_cfg_from_registry(tasks.PLAY_TASK, "rsl_rl_cfg_entry_point")
    cfg.scene.num_envs = args.num_envs
    cfg.seed = args.seed
    with launch_simulation(cfg, args):
        from isaaclab_rl.rsl_rl import (RslRlVecEnvWrapper, create_rsl_rl_runner,
                                        check_rsl_rl_version, handle_deprecated_rsl_rl_cfg)
        env = gym.make(tasks.PLAY_TASK, cfg=cfg).unwrapped
        try:
            agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, check_rsl_rl_version())
            wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
            runner = create_rsl_rl_runner(wrapped, agent_cfg)
            runner.load(str(checkpoint))
            policy = runner.get_inference_policy(device=env.device)
            obs = wrapped.get_observations()
            episodes = []
            with torch.inference_mode():
                while len(episodes) < args.episodes:
                    obs, _, dones, _ = wrapped.step(policy(obs))
                    policy.reset(dones)
                    for i in dones.nonzero(as_tuple=False).flatten().tolist():
                        episodes.append({"success": bool(env.termination_manager.get_term("success")[i]),
                                         "dropped": bool(env.termination_manager.get_term("dropped")[i]),
                                         "time_out": bool(env.termination_manager.get_term("time_out")[i])})
                        if len(episodes) == args.episodes:
                            break
            count = sum(e["success"] for e in episodes)
            output = {"checkpoint": str(checkpoint), "seed": args.seed, "episodes": len(episodes),
                      "successes": count, "success_rate": count / len(episodes),
                      "drops": sum(e["dropped"] for e in episodes),
                      "timeouts": sum(e["time_out"] for e in episodes), "results": episodes}
            directory = ROOT / "evaluation"
            directory.mkdir(exist_ok=True)
            (directory / "latest.json").write_text(json.dumps(output, indent=2))
            print(json.dumps(output, indent=2), flush=True)
        finally:
            env.close()


if __name__ == "__main__":
    main()
