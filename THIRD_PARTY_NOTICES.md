# Sources and acknowledgements

The repository maintainer's contribution is the adaptation of an existing
pick-and-place workflow to the FAIRINO FR5 arm in Isaac Lab, including prepared
robot assets, Cartesian control, the FR5 task, and validation utilities. The
upstream framework, original robot descriptions, and tutorial work remain
credited to their respective authors.

## FR5 robot description — Wang Ziyuan

- Author: **Wang Ziyuan (WangZY233)**, as identified by the upstream README citation.
- Source: [FR5_Reinforcement-learning](https://github.com/WangZY233/FR5_Reinforcement-learning).
- Included material: the selected `fr5_description` URDF, USD layers, and meshes
  under `fr5_preparation/package/fr5_description/`.
- Upstream project: grasping with PyBullet and Stable Baselines3. Its training
  commands and reported results do not describe this Isaac Lab implementation.
- Local source: `D:\FR5_Reinforcement-learning\fr5_description` at preparation time.
- License: upstream repository's Mulan PSL v2 text is preserved verbatim in
  [licenses/FR5-Reinforcement-learning-MulanPSL-2.0.txt](licenses/FR5-Reinforcement-learning-MulanPSL-2.0.txt).
  The copied ROS `package.xml` still contains its original `TODO` license field.

The source hashes are retained in
[`source_manifest.json`](fr5_preparation/package/source_manifest.json).
`fr5_withgripper.original.urdf` preserves the original URDF. Adaptations include
relative mesh paths, revised mass/inertia estimates, gripper collision setup,
portable materials, and an authored `fr5_arm.usda` physics override. Some imported
USD layers also have material/dependency changes; the manifest describes the
original inputs, not the hashes of every modified output.

## Original custom arm — Robot Mania

The retained `cumotion_arm_description/package.xml` identifies **Robot Mania**
as maintainer and declares **Apache License 2.0**. The custom-arm task, gripper
diagnostics, cube asset, and imported USD layers are retained as the earlier
baseline. The previous local README referred to a tutorial but did not identify
its repository or video URL. No unverified tutorial URL or additional authorship
claim has been added. The precise upstream tutorial and asset license text still
need to be recovered before treating attribution for this baseline as complete.

## Isaac Lab Project Developers

- Framework: [Isaac Lab](https://github.com/isaac-sim/IsaacLab).
- Existing source headers credit **The Isaac Lab Project Developers** and specify
  **BSD-3-Clause**; these headers are preserved.
- License copy: [licenses/IsaacLab-BSD-3-Clause.txt](licenses/IsaacLab-BSD-3-Clause.txt).
- NVIDIA Isaac Sim and external runtime dependencies retain their own terms.

## FAIRINO / FAIR-INNOVATION

The prepared FR5 model uses nominal arm-link masses referenced from the
[official FR5 V6 URDF](https://github.com/FAIR-INNOVATION/frcobot_ros2/blob/main/fairino_description/urdf/fairino5_v6.urdf).
Only the mass values are used as a reference. The imported model's frames differ,
so its centers of mass and inertias were estimated from its own meshes. This
does not establish calibration or manufacturer validation.

## Licensing scope

No single repository-wide license has been assigned to all files. Existing
third-party notices and applicable licenses remain in effect. The maintainer
has not yet selected a license for new contributions. Attribution alone does
not replace the applicable license texts or resolve missing upstream provenance.
