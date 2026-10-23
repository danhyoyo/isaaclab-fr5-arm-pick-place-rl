"""Run a bounded gripper test, or display the robot with --viz kit --cycles 0."""
import argparse
import json
from pathlib import Path
import torch
import warp as wp


def main():
    wp.config.enable_backward = False
    from isaaclab.app import add_launcher_args, launch_simulation
    from isaaclab.assets import Articulation
    from isaaclab.utils import clone
    from isaaclab_physx.physics import PhysxCfg
    import isaaclab.sim as sim_utils
    from fr5_cfg import FR5_CFG, GRIPPER_OPEN, GRIPPER_CLOSED, GRIPPER_JOINTS

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cycles", type=int, default=2)
    parser.add_argument("--gravity", action="store_true", help="Include gravity in the joint-motion test.")
    add_launcher_args(parser)
    parser.set_defaults(visualizer="kit")
    args = parser.parse_args()
    if args.cycles < 0:
        parser.error("cycles must be nonnegative")
    cfg = sim_utils.SimulationCfg(dt=0.005, device=args.device, physics=PhysxCfg(),
                                 gravity=(0, 0, -9.81 if args.gravity else 0.0))
    records = []
    with launch_simulation(cfg, args):
        sim = sim_utils.SimulationContext(cfg)
        try:
            light = sim_utils.DomeLightCfg(intensity=2500.0)
            light.func("/World/Light", light)
            robot_cfg = clone(FR5_CFG)
            robot_cfg.prim_path = "/World/Robot"
            robot_cfg.init_state.pos = (0, 0, 0.6)
            robot = Articulation(robot_cfg)
            sim.set_camera_view(eye=(-1.3, -1.0, 1.1), target=(-0.65, -0.2, 0.7))
            sim.reset()
            robot.write_joint_position_to_sim_index(position=robot.data.default_joint_pos.torch.clone())
            robot.write_joint_velocity_to_sim_index(velocity=robot.data.default_joint_vel.torch.clone())
            robot.reset()
            ids = [robot.joint_names.index(name) for name in GRIPPER_JOINTS]
            target = robot.data.default_joint_pos.torch.clone()
            segment = round(2.0 / cfg.dt)
            steps = 0
            while sim.is_running() and (args.cycles == 0 or steps < args.cycles * segment * 2):
                closing = (steps // segment) % 2 == 0
                value = GRIPPER_CLOSED if closing else GRIPPER_OPEN
                target[:, ids] = value
                robot.actuators.target_command.set_position_index(value=target)
                robot.write_data_to_sim()
                sim.step()
                robot.update(cfg.dt)
                steps += 1
                if steps % segment == 0:
                    actual = robot.data.joint_pos.torch[0, ids].cpu().tolist()
                    entry = {"state": "closed" if closing else "open", "target_m": value, "actual_m": actual}
                    records.append(entry)
                    print(json.dumps(entry), flush=True)
                    assert all(abs(q - value) < 0.001 for q in actual), entry
                    assert abs(actual[0] - actual[1]) < 0.001, entry
                    assert torch.isfinite(robot.data.joint_pos.torch).all()
            if args.cycles:
                assert len(records) == args.cycles * 2
                output = {"passed": True, "gravity": args.gravity, "cycles": args.cycles,
                          "joint_names": robot.joint_names, "segments": records,
                          "scope": "isolated open/close motion; not grasp success"}
                root = Path(__file__).resolve().parent / "verification"
                root.mkdir(exist_ok=True)
                (root / ("motion_gravity.json" if args.gravity else "motion.json")).write_text(json.dumps(output, indent=2))
                print("PASS: FR5 gripper motion", flush=True)
        finally:
            sim.stop()
            sim.clear_instance()


if __name__ == "__main__":
    main()
