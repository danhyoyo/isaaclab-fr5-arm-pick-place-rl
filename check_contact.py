"""Check that both jaw colliders stop against a fixed 36 mm block and reopen."""
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
    from fr5_cfg import FR5_CFG

    parser = argparse.ArgumentParser(description=__doc__)
    add_launcher_args(parser)
    args = parser.parse_args()
    cfg = sim_utils.SimulationCfg(dt=0.005, device=args.device, physics=PhysxCfg(), gravity=(0, 0, 0))
    with launch_simulation(cfg, args):
        from pxr import Gf, Usd, UsdGeom, UsdPhysics
        sim = sim_utils.SimulationContext(cfg)
        try:
            robot_cfg = clone(FR5_CFG)
            robot_cfg.prim_path = "/World/Robot"
            robot_cfg.init_state.pos = (0, 0, 0.6)
            robot = Articulation(robot_cfg)
            body = sim.stage.GetPrimAtPath("/World/Robot/hand_base_link")
            body_world = UsdGeom.Xformable(body).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            block = UsdGeom.Cube.Define(sim.stage, "/World/TestBlock")
            block.CreateSizeAttr(0.036)
            block.AddTransformOp().Set(Gf.Matrix4d(1).SetTranslate(Gf.Vec3d(-0.004, 0.00436, 0.122)) * body_world)
            UsdPhysics.CollisionAPI.Apply(block.GetPrim()).CreateCollisionEnabledAttr(True)
            sim.reset()
            robot.write_joint_position_to_sim_index(position=robot.data.default_joint_pos.torch.clone())
            robot.write_joint_velocity_to_sim_index(velocity=robot.data.default_joint_vel.torch.clone())
            robot.reset()
            ids = [robot.joint_names.index(name) for name in ["fj1", "fj2"]]
            target = robot.data.default_joint_pos.torch.clone()
            values = {}
            for label, value in [("close_against_block", 0.04), ("reopen", 0.0)]:
                target[:, ids] = value
                for _ in range(400):
                    robot.actuators.target_command.set_position_index(value=target)
                    robot.write_data_to_sim()
                    sim.step()
                    robot.update(cfg.dt)
                values[label] = robot.data.joint_pos.torch[0, ids].cpu().tolist()
                print(label, values[label], flush=True)
            assert all(0.005 < q < 0.035 for q in values["close_against_block"]), values
            assert all(abs(q) < 0.001 for q in values["reopen"]), values
            output = {"passed": True, "block_size_m": 0.036, "joints": ["fj1", "fj2"],
                      "positions_m": values,
                      "scope": "collision blocking against a fixed box; not a lift/hold or grasp-success test"}
            path = Path(__file__).resolve().parent / "verification"
            path.mkdir(exist_ok=True)
            (path / "contact.json").write_text(json.dumps(output, indent=2))
            print("PASS: both jaws stop at block and reopen", flush=True)
        finally:
            sim.stop()
            sim.clear_instance()


if __name__ == "__main__":
    main()
