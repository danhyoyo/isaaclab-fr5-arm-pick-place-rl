"""Measure gripper motion independently of the learned policy.

uv run --project D:\\IsaacLab --no-sync python check_gripper.py --viz kit
Use --viz none for an automated headless check.
"""

import argparse
import json
import math
import sys

import torch
import warp as wp

from run import PROJECT_ROOT, PLAY_TASK, register_tasks


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    wp.config.enable_backward = False
    register_tasks()
    from isaaclab.app import add_launcher_args, launch_simulation
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
    import gymnasium as gym

    parser = argparse.ArgumentParser(description=__doc__)
    add_launcher_args(parser)
    args = parser.parse_args()
    cfg = load_cfg_from_registry(PLAY_TASK, "env_cfg_entry_point")
    cfg.scene.num_envs = 1
    cfg.seed = 42
    # Keep the arm and cube clear of the floor during the isolated joint test.
    cfg.scene.robot.init_state.pos = (0.0, 0.0, 0.5)
    cfg.episode_length_s = 30.0
    results = {}

    with launch_simulation(cfg, args):
        env = gym.make(PLAY_TASK, cfg=cfg).unwrapped
        try:
            robot = env.scene["robot"]
            joint = robot.joint_names.index("cumotion_arm_joint_hand")
            term = env.action_manager.get_term("gripper_action")
            limits = robot.data.joint_pos_limits.torch[0, joint].cpu().tolist()
            results["joint_limits_deg"] = [math.degrees(x) for x in limits]
            results["configured_close_deg"] = math.degrees(
                cfg.actions.gripper_action.close_command_expr["cumotion_arm_joint_hand"]
            )
            actions = torch.zeros((1, env.action_manager.total_action_dim), device=env.device)
            from isaaclab.utils.math import quat_apply

            def tips():
                # Centers of the outer contact-pad vertices in each body's frame,
                # measured from this robot's collision meshes (meters).
                positions = []
                for name, offset in (
                    ("cumotion_arm_link6", (0.0, -0.02075, 0.010369)),
                    ("cumotion_arm_link_hand", (0.0, 0.006, 0.046)),
                ):
                    body_id = robot.body_names.index(name)
                    point = torch.tensor([offset], device=env.device)
                    world = robot.data.body_pos_w.torch[:, body_id] + quat_apply(
                        robot.data.body_quat_w.torch[:, body_id], point
                    )
                    positions.append(world[0])
                return {
                    "pad_center_distance_mm": torch.linalg.vector_norm(positions[0] - positions[1]).item() * 1000,
                    "pad_centers_world_m": [p.cpu().tolist() for p in positions],
                }
            for label, target in (("negative_60", -math.pi / 3), ("positive_60", math.pi / 3)):
                env.reset()
                term._close_command.fill_(target)
                actions[:, -1] = -1.0
                samples = []
                with torch.inference_mode():
                    for step in range(100):
                        env.step(actions)
                        samples.append(math.degrees(robot.data.joint_pos.torch[0, joint].item()))
                results[label] = {
                    "target_deg": math.degrees(target),
                    "final_deg": samples[-1],
                    "min_deg": min(samples),
                    "max_deg": max(samples),
                    **tips(),
                }
                print(f"GRIPPER {label}: {results[label]}", flush=True)
            actions[:, -1] = 1.0
            with torch.inference_mode():
                for _ in range(100):
                    env.step(actions)
            results["reopened_deg"] = math.degrees(robot.data.joint_pos.torch[0, joint].item())
            results["zero_angle_geometry"] = tips()
            print("GRIPPER RESULTS", json.dumps(results, indent=2), flush=True)
            output = PROJECT_ROOT / "logs/verification/gripper.json"
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(results, indent=2), encoding="utf-8")
            close_deg = results["configured_close_deg"]
            lower, upper = results["joint_limits_deg"]
            if not lower - 0.1 <= close_deg <= upper + 0.1:
                raise AssertionError("Configured gripper close target is outside the USD joint limits")
            if results["negative_60"]["final_deg"] > -45.0:
                raise AssertionError("Gripper did not move toward its close target")
            if results["negative_60"]["pad_center_distance_mm"] >= results["zero_angle_geometry"]["pad_center_distance_mm"] * 0.5:
                raise AssertionError("Close motion did not reduce the distance between the contact pads")
            if abs(results["reopened_deg"]) > 2.0:
                raise AssertionError("Gripper did not reopen")
            print("PASS: target is in range; closing reduces jaw gap and reopening returns to zero", flush=True)
        finally:
            env.close()


if __name__ == "__main__":
    main()
