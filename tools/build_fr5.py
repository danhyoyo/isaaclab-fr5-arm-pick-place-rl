"""Prepare a separate FR5 asset; never edit the source description package."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import xml.etree.ElementTree as ET

import numpy as np
import trimesh
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

SOURCE = Path(r"D:\FR5_Reinforcement-learning\fr5_description")
OFFICIAL = "https://github.com/FAIR-INNOVATION/frcobot_ros2/blob/main/fairino_description/urdf/fairino5_v6.urdf"
# Nominal link masses only. Official V6 frames/inertias do NOT fit the older meshes.
MASSES = dict(zip(
    ["base_link", "shoulder_Link", "upperarm_Link", "forearm_Link", "wrist1_Link", "wrist2_Link", "wrist3_Link"],
    [1.6185, 4.64, 10.08, 2.71, 1.56, 1.56, 0.36],
))
MASSES.update(hand_base_link=1.415640442478951, finger_link1=0.0589543896992846, finger_link2=0.0589543896992846)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(target):
    if target.exists() and any(target.iterdir()):
        raise RuntimeError(f"Destination must be empty: {target}")
    target.mkdir(parents=True, exist_ok=True)
    package = target / "fr5_description"
    package.mkdir()
    shutil.copytree(SOURCE / "meshes", package / "meshes")
    for name in ("package.xml", "CMakeLists.txt"):
        shutil.copy2(SOURCE / name, package / name)
    cmake = package / "CMakeLists.txt"
    cmake.write_text(cmake.read_text().replace("install(DIRECTORY launch\n  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})", "# Launch files omitted from this Isaac Lab asset subset."))
    (package / "urdf").mkdir()
    src_urdf = SOURCE / "urdf/fr5_withgripper.urdf"
    shutil.copy2(src_urdf, package / "urdf/fr5_withgripper.original.urdf")
    assetdir = package / "urdf/fr5_withgripper"
    shutil.copytree(SOURCE / "urdf/fr5_withgripper", assetdir)
    sources = [p for p in (SOURCE / "meshes").rglob("*") if p.is_file()]
    sources += [p for p in (SOURCE / "urdf/fr5_withgripper").rglob("*") if p.is_file()]
    sources += [src_urdf, SOURCE / "package.xml", SOURCE / "CMakeLists.txt"]
    manifest = {str(p.relative_to(SOURCE)): digest(p) for p in sources}
    (target / "source_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # Correct the copied base layer, retaining existing material bindings.
    basepath = assetdir / "configuration/fr5_withgripper_base.usd"
    base = Usd.Stage.Open(str(basepath))
    base.RemovePrim("/fr5_robot/world/visuals")
    for prim in [p for p in base.Traverse() if p.IsA(UsdShade.Material)]:
        if prim:
            material = UsdShade.Material(prim)
            shaderprim = next((c for c in prim.GetChildren() if c.IsA(UsdShade.Shader)), None)
            if shaderprim is None:
                continue
            old = UsdShade.Shader(shaderprim)
            if not shaderprim.HasAttribute("info:mdl:sourceAsset"):
                continue
            color = old.GetInput("diffuse_color_constant").Get()
            for name in ("outputs:mdl:surface", "outputs:mdl:volume", "outputs:mdl:displacement"):
                prim.RemoveProperty(name)
            for name in ("info:mdl:sourceAsset", "info:mdl:sourceAsset:subIdentifier", "outputs:out"):
                shaderprim.RemoveProperty(name)
            shader = old
            shader.GetImplementationSourceAttr().Set("id")
            shader.CreateIdAttr("UsdPreviewSurface")
            shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(color)
            shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.5)
            material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    base.GetRootLayer().Save()

    fixed = assetdir / "fr5_arm.usda"
    layer = Sdf.Layer.CreateNew(str(fixed))
    layer.subLayerPaths = ["fr5_withgripper.usd"]
    layer.Save()
    stage = Usd.Stage.Open(str(fixed))
    stage.SetDefaultPrim(stage.GetPrimAtPath("/fr5_robot"))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    stage.GetRootLayer().customLayerData = {
        "status": "simulation_estimates_not_calibrated",
        "mass_reference": OFFICIAL,
        "inertia_method": "uniform closed convex hull of original collision mesh, normalized to assigned mass",
    }
    cache = UsdGeom.XformCache()
    urdf = ET.parse(src_urdf)
    records = {}
    for name, mass in MASSES.items():
        body = stage.GetPrimAtPath("/fr5_robot/" + name)
        meshprim = next(p for p in Usd.PrimRange(body, Usd.TraverseInstanceProxies())
                        if p.IsA(UsdGeom.Mesh) and "/collisions/" in str(p.GetPath()))
        mesh = UsdGeom.Mesh(meshprim)
        vertices = np.asarray(mesh.GetPointsAttr().Get())
        matrix = np.array(cache.GetLocalToWorldTransform(meshprim) * cache.GetLocalToWorldTransform(body).GetInverse())
        vertices = (np.c_[vertices, np.ones(len(vertices))] @ matrix)[:, :3]
        hull = trimesh.convex.convex_hull(vertices)
        hull.density = mass / hull.volume
        center = hull.center_mass
        inertia = hull.moment_inertia
        values, vectors = np.linalg.eigh(inertia)
        if np.linalg.det(vectors) < 0:
            vectors[:, 0] *= -1
        rotation = Gf.Matrix3d(vectors.T.tolist()).ExtractRotation().GetQuat()
        old = UsdPhysics.MassAPI(body)
        records[name] = {
            "old_mass_kg": old.GetMassAttr().Get(),
            "old_com_m": list(old.GetCenterOfMassAttr().Get()),
            "mass_kg": mass, "com_m": center.tolist(),
            "inertia_kg_m2": inertia.tolist(), "principal_inertia": values.tolist(),
            "geometry_bounds_m": [vertices.min(0).tolist(), vertices.max(0).tolist()],
            "method": "uniform convex hull estimate; not manufacturer inertia",
        }
        api = UsdPhysics.MassAPI.Apply(body)
        api.CreateMassAttr(mass)
        api.CreateCenterOfMassAttr(Gf.Vec3f(*center))
        api.CreateDiagonalInertiaAttr(Gf.Vec3f(*values))
        api.CreatePrincipalAxesAttr(Gf.Quatf(rotation))
        inertial = urdf.find(f"./link[@name='{name}']/inertial")
        inertial.find("mass").set("value", str(mass))
        inertial.find("origin").set("xyz", " ".join(f"{x:.12g}" for x in center))
        inertial.find("origin").set("rpy", "0 0 0")
        for key, ij in {"ixx": (0, 0), "iyy": (1, 1), "izz": (2, 2), "ixy": (0, 1), "ixz": (0, 2), "iyz": (1, 2)}.items():
            inertial.find("inertia").set(key, f"{inertia[ij]:.12g}")

    # Explicit force drives: linear values N/m and N*s/m; angular values N*m/rad.
    for name in [f"j{i}" for i in range(1, 7)] + ["fj1", "fj2"]:
        prim = stage.GetPrimAtPath("/fr5_robot/joints/" + name)
        finger = name.startswith("f")
        drive = UsdPhysics.DriveAPI.Apply(prim, "linear" if finger else "angular")
        stiffness = 2000.0 if finger else (2000.0 if name in ["j1", "j2", "j3"] else 500.0)
        damping = 20.0 if finger else (100.0 if name in ["j1", "j2", "j3"] else 25.0)
        drive.CreateTypeAttr("force")
        drive.CreateStiffnessAttr(stiffness)
        drive.CreateDampingAttr(damping)
        drive.CreateMaxForceAttr(40.0 if finger else (150.0 if name in ["j1", "j2", "j3"] else 28.0))
        drive.CreateTargetPositionAttr(0.0)
        drive.CreateTargetVelocityAttr(0.0)
        if finger:
            prim.GetAttribute("physxJoint:maxJointVelocity").Set(0.1)
            urdf.find(f"./joint[@name='{name}']/limit").set("velocity", "0.1")
        else:
            urdf.find(f"./joint[@name='{name}']/limit").set("effort", "150" if name in ["j1", "j2", "j3"] else "28")

    # Author collision on actual gripper mesh prims and preserve the concave jaw shape.
    material = UsdShade.Material.Define(stage, "/fr5_robot/Looks/GripperContact")
    physicsmat = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    physicsmat.CreateStaticFrictionAttr(0.8)
    physicsmat.CreateDynamicFrictionAttr(0.6)
    physicsmat.CreateRestitutionAttr(0.0)
    for name in ["hand_base_link", "finger_link1", "finger_link2"]:
        container = stage.GetPrimAtPath("/fr5_robot/" + name + "/collisions")
        container.SetInstanceable(False)
        for prim in list(Usd.PrimRange(container)):
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(False)
            if prim.IsA(UsdGeom.Mesh):
                UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr(True)
                UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr("convexDecomposition")
                UsdShade.MaterialBindingAPI.Apply(prim).Bind(material, materialPurpose="physics")

    root = stage.GetPrimAtPath("/fr5_robot/root_joint")
    root.GetAttribute("physxArticulation:solverPositionIterationCount").Set(16) if root.HasAttribute("physxArticulation:solverPositionIterationCount") else root.CreateAttribute("physxArticulation:solverPositionIterationCount", Sdf.ValueTypeNames.Int).Set(16)
    root.CreateAttribute("physxArticulation:solverVelocityIterationCount", Sdf.ValueTypeNames.Int).Set(4)
    stage.GetRootLayer().Save()
    # Portable relative paths for direct URDF import without ROS package discovery.
    for mesh in urdf.findall(".//mesh"):
        mesh.set("filename", "../" + mesh.get("filename").split("package://fr5_description/")[1])
    for j in urdf.findall("joint"):
        if j.get("type") == "fixed":
            for tag in ["limit", "axis"]:
                element = j.find(tag)
                if element is not None:
                    j.remove(element)
    ET.indent(urdf)
    urdf.write(package / "urdf/fr5_withgripper.urdf", encoding="utf-8", xml_declaration=True)
    report = {"status": "simulation_estimates_not_calibrated", "mass_reference": OFFICIAL,
              "total_mass_kg": sum(MASSES.values()), "links": records,
              "gripper": {"open_m": 0.0, "closed_m": 0.04, "stiffness_N_per_m": 2000,
                          "damping_Ns_per_m": 20, "max_force_N": 40, "max_velocity_m_per_s": 0.1},
              "notes": ["Original geometry differs from official FR5 V6; only nominal masses transferred.",
                        "COM and inertia are uniform convex hull estimates, not measured properties.",
                        "Gripper source masses retained; contact friction and drive gains are simulation starting values.",
                        "Both sliders must receive the same target; no mimic joint added.",
                        "URDF is a corrected source; use fr5_arm.usda for authored drives/material/collision setup."]}
    (target / "parameters.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Prepared {fixed}; total assigned mass = {sum(MASSES.values()):.6f} kg")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("target", type=Path)
    build(p.parse_args().target.resolve())
