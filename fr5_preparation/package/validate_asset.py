"""Check composed asset dependencies, inertia, joints and jaw geometry without a simulator."""
from pathlib import Path
import hashlib
import json
import xml.etree.ElementTree as ET
import numpy as np
from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdUtils

ROOT = Path(__file__).resolve().parent
ASSET = ROOT / "fr5_description/urdf/fr5_withgripper/fr5_arm.usda"


def frame(joint, index):
    matrix = Gf.Matrix4d(1).SetRotate(Gf.Rotation(joint.GetAttribute(f"physics:localRot{index}").Get()))
    matrix.SetTranslateOnly(Gf.Vec3d(joint.GetAttribute(f"physics:localPos{index}").Get()))
    return matrix


def validate():
    stage = Usd.Stage.Open(str(ASSET))
    layers, assets, unresolved = UsdUtils.ComputeAllDependencies(str(ASSET))
    assert not unresolved, unresolved
    for layer in layers:
        assert Path(layer.realPath).resolve().is_relative_to(ROOT), layer.identifier
    assert UsdGeom.GetStageMetersPerUnit(stage) == 1.0
    assert UsdGeom.GetStageUpAxis(stage) == "Z"
    assert stage.GetPrimAtPath("/fr5_robot/root_joint").HasAPI(UsdPhysics.ArticulationRootAPI)
    movable = [p.GetName() for p in stage.Traverse()
               if p.IsA(UsdPhysics.RevoluteJoint) or p.IsA(UsdPhysics.PrismaticJoint)]
    assert set(movable) == {"j1", "j2", "j3", "j4", "j5", "j6", "fj1", "fj2"}, movable
    data = json.loads((ROOT / "parameters.json").read_text())
    urdf = ET.parse(ROOT / "fr5_description/urdf/fr5_withgripper.urdf")
    for mesh in urdf.findall(".//mesh"):
        assert (ROOT / "fr5_description/urdf" / mesh.get("filename")).is_file()
    for name, record in data["links"].items():
        mass = UsdPhysics.MassAPI(stage.GetPrimAtPath("/fr5_robot/" + name))
        assert abs(mass.GetMassAttr().Get() - record["mass_kg"]) < 1e-5
        com = np.array(mass.GetCenterOfMassAttr().Get())
        low, high = np.array(record["geometry_bounds_m"])
        assert np.all(com >= low - 1e-6) and np.all(com <= high + 1e-6), name
        principal = np.array(mass.GetDiagonalInertiaAttr().Get())
        assert np.all(principal > 0) and 2 * principal.max() <= principal.sum() + 1e-6
        rotation = np.array(Gf.Matrix3d(mass.GetPrincipalAxesAttr().Get()))
        tensor = rotation.T @ np.diag(principal) @ rotation
        assert np.allclose(tensor, record["inertia_kg_m2"], atol=1e-6), name
        inertial = urdf.find(f"./link[@name='{name}']/inertial")
        assert np.allclose(np.fromstring(inertial.find("origin").get("xyz"), sep=" "), com, atol=1e-6)
    cache = UsdGeom.XformCache()
    gap = {}
    for target in [0.0, 0.02, 0.04]:
        tips = []
        for index in [1, 2]:
            body = stage.GetPrimAtPath(f"/fr5_robot/finger_link{index}")
            joint = stage.GetPrimAtPath(f"/fr5_robot/joints/fj{index}")
            mesh = next(p for p in Usd.PrimRange(body, Usd.TraverseInstanceProxies())
                        if p.IsA(UsdGeom.Mesh) and "/collisions/" in str(p.GetPath()))
            assert UsdPhysics.CollisionAPI(mesh).GetCollisionEnabledAttr().Get()
            assert UsdPhysics.MeshCollisionAPI(mesh).GetApproximationAttr().Get() == "convexDecomposition"
            movement = Gf.Matrix4d(1).SetTranslate(Gf.Vec3d(target, 0, 0))
            matrix = (cache.GetLocalToWorldTransform(mesh) * cache.GetLocalToWorldTransform(body).GetInverse()
                      * frame(joint, 1).GetInverse() * movement * frame(joint, 0))
            vertices = np.asarray(UsdGeom.Mesh(mesh).GetPointsAttr().Get())
            points = (np.c_[vertices, np.ones(len(vertices))] @ np.array(matrix))[:, :3]
            tips.append(points[points[:, 2] > 0.115])
        gap[str(target)] = float(tips[1][:, 0].min() - tips[0][:, 0].max()) * 1000
    assert 79 < gap["0.0"] < 81
    assert 39 < gap["0.02"] < 41
    assert 0 <= gap["0.04"] < 1
    output = {"passed": True, "dependencies": len(layers), "unresolved": unresolved,
              "movable_joints": movable, "jaw_tip_gap_mm": gap,
              "total_assigned_mass_kg": data["total_mass_kg"],
              "scope": "static geometry and parameter consistency; not a grasp-success evaluation"}
    (ROOT / "verification").mkdir(exist_ok=True)
    (ROOT / "verification/static.json").write_text(json.dumps(output, indent=2))
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    validate()
