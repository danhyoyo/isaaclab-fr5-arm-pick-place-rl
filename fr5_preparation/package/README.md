# Prepared FR5 asset for Isaac Lab

This package contains the selected FR5-with-gripper description adapted from
**Wang Ziyuan / WangZY233's**
[FR5_Reinforcement-learning](https://github.com/WangZY233/FR5_Reinforcement-learning).
See [third-party notices](../../THIRD_PARTY_NOTICES.md) for attribution, license
texts, and the distinction between source inputs and modified assets.

## Entry asset and layout

Open `fr5_description/urdf/fr5_withgripper/fr5_arm.usda`. This layer authors the
physics overrides on the imported `fr5_withgripper.usd`. Keep the full
`fr5_description/` tree, including its configuration layers and meshes. The base
imported USD alone does not include all corrections. Git LFS must download the
actual USD and mesh objects before use.

`fr5_cfg.py` provides `FR5_CFG` to Isaac Lab. The RL task lives in
`../../fr5_rl_development/`; its launcher resolves this package automatically.
This package alone is a robot asset configuration, not an RL environment.

## Preparation changes

- Nominal seven arm-link masses referenced from the
  [official FAIRINO FR5 V6 URDF](https://github.com/FAIR-INNOVATION/frcobot_ros2/blob/main/fairino_description/urdf/fairino5_v6.urdf):
  1.6185, 4.64, 10.08, 2.71, 1.56, 1.56, 0.36 kg.
- Original gripper body mass retained at approximately 1.4156404425 kg and each
  finger at 0.0589543897 kg; total assigned mass approximately 24.06205 kg.
- Centers of mass and inertia estimated from closed collision-mesh convex hulls,
  assuming uniform density and normalized to the assigned mass. URDF and USD use
  matching tensors; USD stores principal inertia and principal axes.
- Finger drives: stiffness 2000 N/m, damping 20 N·s/m, maximum force 40 N,
  maximum speed 0.1 m/s. Both joints open at 0 m and close toward 0.04 m.
- Base arm/wrist drives: stiffness/damping 2000/100 and 500/25; effort limits
  150 N·m and 28 N·m. The RL scene overrides stiffness/damping for gravity loads.
- Gripper mesh colliders use convex decomposition; static/dynamic friction 0.8/0.6,
  restitution 0. Self-collision is disabled.
- Removed an empty world-image reference; converted OmniPBR to UsdPreviewSurface
  to remove external MDL dependencies. Diffuse color is retained; appearance may differ.
- Corrected URDF mesh paths to relative paths and removed redundant fixed-joint
  axis/limit elements. Simulator-specific drives/materials do not transfer via URDF.

These are **simulation estimates, not a calibrated real-robot dynamics model**.
The imported model has different frames from the official V6 description, so official
V6 centers of mass/inertias were not copied into the older geometry. Hull estimates
omit internal cavities, motors, gearboxes, and actual material distribution.

## Validation commands

From the repository root, with the Isaac Lab interpreter configured:

```powershell
& $IsaacPython .\fr5_preparation\package\validate_asset.py
& $IsaacPython .\fr5_preparation\package\gripper_scene.py --viz none --cycles 2
& $IsaacPython .\fr5_preparation\package\gripper_scene.py --viz none --cycles 2 --gravity
& $IsaacPython .\fr5_preparation\package\check_contact.py --viz none
& $IsaacPython .\fr5_preparation\package\gripper_scene.py --viz kit --cycles 0
```

The gripper-only scene raises the robot by 0.6 m and disables gravity by default;
`--gravity` enables it. A cycle count of zero displays continuous open/close motion.
Finite checks write records into `verification/`.

## Provenance and retained checks

- `parameters.json`: parameter values, estimation method, and original/revised data.
- `source_manifest.json`: SHA-256 hashes of original preparation inputs.
- `fr5_description/urdf/fr5_withgripper.original.urdf`: original URDF for comparison.
- `verification/static.json`: composed dependencies, joints, inertia, and jaw geometry.
- `verification/motion*.json`: finite finger motion checks.
- `verification/contact.json`: obstruction by a fixed 36 mm cube followed by reopening.

Retained local records report approximately 80.08 / 40.08 / 0.08 mm tip gaps with
both finger coordinates at 0 / 0.02 / 0.04 m. The contact check reports stopping
near 0.02068 / 0.02048 m against the fixed cube. These checks do not measure grip
force or prove a successful lift of a free object. Use the FR5 demo and learned-policy
evaluation for their respective manipulation checks.

## Rebuilding

Normal use requires no rebuild. `tools/build_fr5.py` is a historical preparation
utility with a machine-specific `SOURCE` path. Rebuilding requires the upstream
source checkout, the Isaac Lab USD bindings, NumPy, and trimesh. Inspect and update
its source path first, and choose an empty destination. Do not run it over this
prepared package or over the upstream source.

The [previous asset README](../../docs/archive/ASSET.original.vi.md) is retained
unchanged for historical details.
