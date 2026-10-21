"""Show a stationary robot opening and closing its gripper, without an RL task.

uv run --project D:\\IsaacLab --no-sync python gripper_scene.py
"""

import argparse
import math
import sys
import time

import warp as wp

from run import register_tasks


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    wp.config.enable_backward = False
    register_tasks()

    from isaaclab.app import add_launcher_args, launch_simulation
    from isaaclab.assets import Articulation
    from isaaclab.utils import clone
    from isaaclab_physx.physics import PhysxCfg
    import isaaclab.sim as sim_utils
    from pick_and_place.custom_arm.joint_pos_place_env_cfg import (
        CUSTOM_ARM_CFG, GRIPPER_OPEN_ANGLE, GRIPPER_CLOSED_ANGLE,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cycles", type=int, default=0, help="Close/open cycles; 0 repeats until the window closes.")
    add_launcher_args(parser)
    parser.set_defaults(visualizer="kit")
    args = parser.parse_args()
    if args.cycles < 0:
        parser.error("--cycles must be non-negative")

    sim_cfg = sim_utils.SimulationCfg(
        dt=0.01, device=args.device, physics=PhysxCfg(), gravity=(0.0, 0.0, 0.0),
    )
    with launch_simulation(sim_cfg, args):
        sim = sim_utils.SimulationContext(sim_cfg)
        try:
            ground = sim_utils.GroundPlaneCfg()
            ground.func("/World/Ground", ground)
            light = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.8, 0.8, 0.8))
            light.func("/World/Light", light)

            robot_cfg = clone(CUSTOM_ARM_CFG)
            robot_cfg.prim_path = "/World/Robot"
            robot = Articulation(robot_cfg)
            sim.set_camera_view(eye=(0.45, -0.40, 0.38), target=(0.18, -0.01, 0.22))
            sim.reset()
            robot.write_joint_position_to_sim_index(position=robot.data.default_joint_pos.torch.clone())
            robot.write_joint_velocity_to_sim_index(velocity=robot.data.default_joint_vel.torch.clone())
            robot.reset()
            target = robot.data.default_joint_pos.torch.clone()
            hand = robot.joint_names.index("cumotion_arm_joint_hand")
            half_cycle = round(2.0 / sim_cfg.dt)
            steps = 0
            angles = {"CLOSE": [], "OPEN": []}
            print("Gripper scene: CLOSE -60 deg for 2 s, OPEN 0 deg for 2 s. Ctrl+C to stop.")

            while sim.is_running() and (args.cycles == 0 or steps < args.cycles * 2 * half_cycle):
                started = time.perf_counter()
                closing = (steps // half_cycle) % 2 == 0
                label = "CLOSE" if closing else "OPEN"
                if steps % half_cycle == 0:
                    angle_target = GRIPPER_CLOSED_ANGLE if closing else GRIPPER_OPEN_ANGLE
                    target[:, hand] = angle_target
                    print(f"{label}: target={math.degrees(angle_target):.1f} deg")
                # Hold every arm joint at its default pose; move only the hand.
                robot.actuators.target_command.set_position_index(value=target)
                robot.write_data_to_sim()
                sim.step()
                robot.update(sim_cfg.dt)
                steps += 1
                if steps % half_cycle == 0:
                    angle = math.degrees(robot.data.joint_pos.torch[0, hand].item())
                    angles[label].append(angle)
                    print(f"{label}: measured={angle:.2f} deg")
                remaining = sim_cfg.dt - (time.perf_counter() - started)
                if remaining > 0:
                    time.sleep(remaining)

            if args.cycles and steps == args.cycles * 2 * half_cycle:
                if not all(abs(angle - math.degrees(GRIPPER_CLOSED_ANGLE)) < 2 for angle in angles["CLOSE"]):
                    raise AssertionError(f"Gripper failed to close: {angles}")
                if not all(abs(angle) < 2 for angle in angles["OPEN"]):
                    raise AssertionError(f"Gripper failed to open: {angles}")
                print(f"PASS: completed {args.cycles} close/open cycles")
        except KeyboardInterrupt:
            pass
        finally:
            sim.stop()
            sim.clear_instance()


if __name__ == "__main__":
    main()
