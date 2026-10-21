"""Physical scripted pick/place check using the same four actions as PPO (no object teleporting)."""
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
    from fr5_pick_place import mdp
    from fr5_pick_place.constants import ACTION_STEP, REST_HEIGHT, ROBOT_BASE_HEIGHT
    import gymnasium as gym
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num_envs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    add_launcher_args(parser)
    args = parser.parse_args()
    cfg = load_cfg_from_registry(tasks.PLAY_TASK, "env_cfg_entry_point")
    cfg.scene.num_envs = args.num_envs
    cfg.seed = args.seed
    records = []
    with launch_simulation(cfg, args):
        env = gym.make(tasks.PLAY_TASK, cfg=cfg).unwrapped
        try:
            env.reset()
            tcp, _, pick, goal, *_ = mdp.geometry(env)
            pick, goal = pick.clone(), goal.clone()
            print("WAYPOINTS", json.dumps({"pick": pick.cpu().tolist(), "goal": goal.cpu().tolist()}), flush=True)
            success = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
            max_height = pick[:, 2].clone()
            phases = [("approach", pick, 0.10, 1, 70), ("descend", pick, 0.0, 1, 75),
                      ("close", pick, 0.0, -1, 45), ("lift", pick, 0.14, -1, 70),
                      ("transfer", goal, 0.14, -1, 90), ("lower", goal, 0.0, -1, 75),
                      ("release", goal, 0.0, 1, 45), ("withdraw", goal, 0.14, 1, 75)]
            with torch.inference_mode():
                for label, location, height, gripper, count in phases:
                    desired = location.clone()
                    desired[:, 2] = REST_HEIGHT + height
                    for _ in range(count):
                        tcp, *_ = mdp.geometry(env)
                        action = torch.zeros(env.num_envs, 4, device=env.device)
                        command = env.action_manager.get_term("arm").target_position.clone()
                        command[:, 2] += ROBOT_BASE_HEIGHT
                        action[:, :3] = ((desired - command) / ACTION_STEP).clamp(-1, 1)
                        action[:, 3] = gripper
                        obs, reward, terminated, truncated, info = env.step(action)
                        assert torch.isfinite(obs["policy"]).all() and torch.isfinite(reward).all()
                        success |= env.termination_manager.get_term("success")
                        assert not env.termination_manager.get_term("dropped").any(), f"Object dropped during {label}"
                        max_height = torch.maximum(max_height, mdp.geometry(env)[2][:, 2])
                    tcp, _, pos, _, fingers, _ = mdp.geometry(env)
                    record = {"phase": label, "tcp_local_m": tcp.cpu().tolist(),
                              "object_local_m": pos.cpu().tolist(), "fingers_m": fingers.cpu().tolist(),
                              "ever_lifted": mdp.progress(env).lifted.cpu().tolist(), "successes": success.cpu().tolist()}
                    records.append(record)
                    print(json.dumps(record), flush=True)
            output = {"passed": bool(success.all()), "controller": "scripted Cartesian waypoints, not a learned policy",
                      "seed": args.seed, "num_envs": env.num_envs, "successes": success.cpu().tolist(),
                      "max_object_height_m": max_height.cpu().tolist(), "phases": records}
            directory = ROOT / "verification"
            directory.mkdir(exist_ok=True)
            (directory / f"scripted_pick_place_seed{args.seed}.json").write_text(json.dumps(output, indent=2))
            assert success.all(), "Scripted physical pick/place did not succeed in every environment"
            print(f"PASS: physical pick/place in {env.num_envs} environments", flush=True)
        finally:
            env.close()


if __name__ == "__main__":
    main()
