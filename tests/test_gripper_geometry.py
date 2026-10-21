"""Static regression checks against the authored robot geometry (no simulator)."""

from pathlib import Path
import sys
import unittest

import numpy as np
from pxr import Gf, Usd, UsdGeom

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run import register_tasks

register_tasks()
from pick_and_place.custom_arm.joint_pos_place_env_cfg import (
    CUSTOM_ARM_CFG, GRIPPER_OPEN_ANGLE, GRIPPER_CLOSED_ANGLE,
)


class GripperGeometryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.stage = Usd.Stage.Open(CUSTOM_ARM_CFG.spawn.usd_path)
        cls.joint = cls.stage.GetPrimAtPath("/cumotion_arm/joints/cumotion_arm_joint_hand")
        cls.parent = cls.stage.GetPrimAtPath("/cumotion_arm/cumotion_arm_link6")
        cls.child = cls.stage.GetPrimAtPath("/cumotion_arm/cumotion_arm_link_hand")

    def frame(self, index):
        frame = Gf.Matrix4d(1)
        frame.SetRotate(Gf.Rotation(Gf.Quatd(self.joint.GetAttribute(f"physics:localRot{index}").Get())))
        frame.SetTranslateOnly(Gf.Vec3d(self.joint.GetAttribute(f"physics:localPos{index}").Get()))
        return frame

    def pad(self, body, moving=False):
        mesh_prim = next(p for p in Usd.PrimRange(body) if p.GetTypeName() == "Mesh")
        points = np.unique(np.round(np.asarray(UsdGeom.Mesh(mesh_prim).GetPointsAttr().Get()), 6), axis=0)
        # Select the small distal contact plates, excluding the servo housing.
        mask = points[:, 2] > 0.04 if moving else ((points[:, 1] < -0.018) & (points[:, 2] > 0))
        self.assertGreater(int(mask.sum()), 0)
        mesh_to_body = (
            UsdGeom.Xformable(mesh_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            * UsdGeom.Xformable(body).ComputeLocalToWorldTransform(Usd.TimeCode.Default()).GetInverse()
        )
        return mesh_to_body.Transform(Gf.Vec3d(*(float(x) for x in points[mask].mean(axis=0))))

    def gap(self, angle):
        parent_world = UsdGeom.Xformable(self.parent).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        rotation = Gf.Matrix4d(1).SetRotate(Gf.Rotation(Gf.Vec3d(1, 0, 0), np.degrees(angle)))
        child_world = self.frame(1).GetInverse() * rotation * self.frame(0) * parent_world
        fixed_tip = parent_world.Transform(self.pad(self.parent))
        moving_tip = child_world.Transform(self.pad(self.child, moving=True))
        return (fixed_tip - moving_tip).GetLength()

    def test_targets_within_joint_limits(self):
        lower = self.joint.GetAttribute("physics:lowerLimit").Get()
        upper = self.joint.GetAttribute("physics:upperLimit").Get()
        for angle in (GRIPPER_OPEN_ANGLE, GRIPPER_CLOSED_ANGLE):
            self.assertGreaterEqual(np.degrees(angle), lower - 1e-5)
            self.assertLessEqual(np.degrees(angle), upper + 1e-5)

    def test_closing_moves_contact_plates_together(self):
        opened = self.gap(GRIPPER_OPEN_ANGLE)
        closed = self.gap(GRIPPER_CLOSED_ANGLE)
        self.assertLess(closed, opened * 0.5)
        print(f"Contact-pad center distance: open={opened * 1000:.1f} mm, closed={closed * 1000:.1f} mm")

    def test_joint_frames_preserve_zero_pose(self):
        parent_world = UsdGeom.Xformable(self.parent).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        authored_child = UsdGeom.Xformable(self.child).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        predicted_child = self.frame(1).GetInverse() * self.frame(0) * parent_world
        np.testing.assert_allclose(np.asarray(predicted_child), np.asarray(authored_child), atol=1e-6)


if __name__ == "__main__":
    unittest.main()
