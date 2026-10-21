# FR5 Pick and Place with Isaac Lab

Reinforcement learning for pick-and-place manipulation with the **FAIRINO FR5**
six-axis arm and a two-finger parallel gripper, using **NVIDIA Isaac Lab**, PhysX,
and **PPO / RSL-RL**.

This repository extends an existing custom-arm pick-and-place project. **The
maintainer's main contribution is adapting the workflow to the FR5 arm**, including
robot asset preparation, Cartesian end-effector control, an FR5-specific task,
and validation utilities. The earlier custom-arm baseline is retained for reference.
Original framework and asset authors are credited in [Acknowledgements](#acknowledgements)
and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

> **Project status:** simulation and training utilities are available. Earlier local
> checks recorded a successful scripted demonstration and a short PPO smoke run;
> these do not establish a trained policy's success rate. No trained FR5 checkpoint
> is distributed with this repository. Robot dynamics are simulation estimates.

## Contents

- [Features and contribution](#features-and-contribution)
- [Repository structure](#repository-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Training, playback, and evaluation](#training-playback-and-evaluation)
- [Task specification](#task-specification)
- [Validation](#validation)
- [Original custom-arm baseline](#original-custom-arm-baseline)
- [Troubleshooting](#troubleshooting)
- [Publishing and contributing](#publishing-and-contributing)
- [Acknowledgements](#acknowledgements)
- [License](#license)

## Features and contribution

- **FR5 integration:** six arm joints (`j1`–`j6`) and two independently simulated
  sliding fingers (`fj1`, `fj2`), with a prepared USD physics override.
- **Asset preparation:** relative dependencies, mesh-based inertia estimates,
  revised mass parameters, portable materials, and gripper collision setup.
- **Cartesian control:** three incremental TCP position actions with differential
  inverse kinematics and a binary open/close action; fixed downward orientation.
- **FR5 pick-and-place task:** randomized cube and target positions, staged rewards,
  episode history, and success conditions requiring lifting, release, and withdrawal.
- **PPO workflow:** configuration checks, training, checkpoint playback, bounded
  playback, and episode-based evaluation.
- **Diagnostics:** static asset checks, finger motion/contact checks, a physical
  waypoint demonstration, and task-logic tests.

The FR5 work builds on upstream assets and Isaac Lab; it does not claim authorship
of the original robot meshes, simulator, framework, or tutorial.

## Repository structure

```text
pick_and_place/
├── README.md
├── THIRD_PARTY_NOTICES.md        # Upstream authors, sources, and licensing scope
├── CONTRIBUTING.md
├── .gitignore                   # Excludes logs, checkpoints, caches, local settings
├── .gitattributes               # Git LFS rules for USD and mesh assets
├── licenses/                    # Preserved upstream license texts
├── docs/archive/                # Previous Vietnamese documentation, unchanged
├── fr5_rl_development/          # Main FR5 reinforcement-learning workflow
│   ├── run.py                   # check / train / play; resolves prepared assets
│   ├── demo.py                  # Scripted physical pick/place demonstration
│   ├── evaluate.py              # Evaluate a learned policy over complete episodes
│   ├── TRAINING.md              # Detailed FR5 training guide
│   ├── fr5_pick_place/
│   │   ├── __init__.py          # Gymnasium task registrations
│   │   ├── constants.py         # Geometry, workspace, poses, success thresholds
│   │   ├── env_cfg.py           # Scene, observations, rewards, resets, terminations
│   │   ├── actions_cfg.py
│   │   ├── actions.py           # Cartesian target integration and differential IK
│   │   ├── mdp.py               # Tensor-based task observations and rewards
│   │   ├── task_logic.py        # Episode history and placement conditions
│   │   └── ppo_cfg.py           # PPO actor/critic and runner settings
│   └── tests/                  # Logic tests and simulation diagnostics
├── fr5_preparation/
│   ├── build_fr5.py             # Historical preparation utility
│   └── package/
│       ├── README.md           # Asset parameters and validation guide
│       ├── fr5_cfg.py          # Shared FR5 articulation configuration
│       ├── fr5_description/    # URDF, USD layers, and meshes; keep together
│       ├── parameters.json     # Mass/inertia estimation records
│       ├── source_manifest.json # SHA-256 hashes of original source inputs
│       ├── validate_asset.py
│       ├── gripper_scene.py
│       ├── check_contact.py
│       ├── tools/build_fr5.py
│       └── verification/       # Retained asset verification records
├── run.py                      # Earlier custom-arm launcher, not the FR5 launcher
├── lift_place_env_cfg.py        # Earlier custom-arm scene/task
├── custom_arm/                 # Earlier robot assets and agent configurations
├── cumotion_arm_description/    # Original custom-arm ROS description
├── cube_5cm/                    # Cube asset for the earlier baseline
├── mdp/                        # Earlier custom-arm task functions
├── check_gripper.py
├── gripper_scene.py
├── tests/                      # Earlier gripper/playback checks
└── how_to_run.txt               # Short command reference
```

Generated FR5 runs are saved under `fr5_rl_development/logs/rsl_rl/fr5_pick_place/`;
evaluation JSON goes to `fr5_rl_development/evaluation/`. Both are ignored by Git.
The FR5 cube is generated in code and does not use the baseline's `cube_5cm` asset.

## Requirements

The following versions were inspected in the development environment. They are
an environment reference, not a portable dependency lockfile.

| Component | Development environment |
| --- | --- |
| Operating system | Windows; commands below use PowerShell |
| Python | 3.12.15 |
| Isaac Lab checkout `VERSION` | 3.0.0 |
| Installed `isaaclab` distribution metadata | 27.2.0 |
| Isaac Sim | 6.2.0.0 |
| RSL-RL (`rsl-rl-lib`) | 5.5.1 |
| PyTorch | 2.11.0+cu130 |
| Gymnasium | 1.3.0 |
| Warp | 1.17.0 |
| Physics | PhysX |

Use an NVIDIA GPU and driver supported by your Isaac Sim release. GPU memory needs
depend on the number of environments; start with eight environments before scaling.
The launchers use `isaaclab_rl.entrypoints` and `isaaclab_physx` from the installed
checkout. Compatibility with Isaac Lab 2.x or other 3.x checkouts is not established.
Linux has not been verified for this repository.

## Installation

### 1. Set up Isaac Lab and its matching Isaac Sim runtime

Install a compatible Isaac Lab / Isaac Sim environment using the
[official Isaac Lab installation documentation](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html).
Select the instructions matching your release; the version table above describes
this project's local environment and does not mean that the latest default branch
will expose the same APIs.

The environment must include Isaac Lab, Isaac Sim, its PhysX backend, the Isaac Lab
RL/task packages, RSL-RL, Torch, Warp, Gymnasium, and NumPy. Asset rebuilding also
uses `trimesh` and the runtime's `pxr` USD bindings. A generic `pip install isaaclab`
or an arbitrary Torch installation is not a substitute for that environment.

This task runs outside the Isaac Lab source tree. No project installation,
replacement of framework files, or additional virtual environment is required.

### 2. Obtain this repository and its large assets

Install [Git LFS](https://git-lfs.com/) before cloning. Replace the example URL
with your fork URL if you are using a fork:

```powershell
git lfs install
git clone https://github.com/danhyoyo/isaaclab-fr5-pick-place-rl.git pick_and_place
Set-Location .\pick_and_place
git lfs pull
```

Keep all mesh files, textures, USD configuration layers, and override layers.
An LFS pointer without its downloaded object cannot be opened as a robot asset.

### 3. Select the Isaac Lab Python interpreter

Run the following from the repository root. Adjust the interpreter location:

```powershell
$IsaacPython = "D:\IsaacLab\.venv\Scripts\python.exe"
& $IsaacPython -c "import torch, warp, gymnasium, isaaclab, isaaclab_physx, isaaclab_rl; print('Runtime imports OK')"
& $IsaacPython .\fr5_rl_development\run.py check
```

The FR5 launcher locates `fr5_preparation/package/` relative to the repository,
so no asset copy or machine-specific `PYTHONPATH` is needed. The configuration check
validates both FR5 task registrations and the main robot asset path without
starting a simulation. Run the static asset check below to validate dependencies.

If you already manage Isaac Lab with `uv`, the equivalent invocation is:

```powershell
uv run --project D:\IsaacLab --no-sync python .\fr5_rl_development\run.py check
```

`--no-sync` reuses the installed environment. It does not install missing packages.

## Quick start

All commands below assume the repository root and `$IsaacPython` defined above.

```powershell
# Verify the composed asset and robot parameters
& $IsaacPython .\fr5_preparation\package\validate_asset.py

# Run a physical waypoint demonstration in a simulation window
& $IsaacPython .\fr5_rl_development\demo.py --viz kit --num_envs 1

# Exercise the training pipeline with a small headless run
& $IsaacPython .\fr5_rl_development\run.py train --num_envs 8 --max_iterations 2 --headless --run_name smoke

# Play the latest checkpoint generated by this FR5 experiment
& $IsaacPython .\fr5_rl_development\run.py play --num_envs 4 --checkpoint latest
```

The demo controls the same four actions as the RL policy and moves the cube through
physical contact after the initial reset. It is a scripted controller, not a learned
policy. The short training command checks the pipeline and is insufficient to learn
the task. Close the viewer or press Ctrl+C to stop interactive playback.

## Training, playback, and evaluation

### Start a new training run

```powershell
& $IsaacPython .\fr5_rl_development\run.py train --num_envs 128 --max_iterations 5000 --headless --run_name fr5_cube30
```

Increase `--num_envs` after checking VRAM and throughput. Use `--viz kit` instead
of `--headless` to train with a viewer; do not supply both flags.

### Play or resume a specific checkpoint

Replace `YOUR_RUN/model_4999.pt` with an actual checkpoint created by training:

```powershell
$Checkpoint = (Resolve-Path ".\fr5_rl_development\logs\rsl_rl\fr5_pick_place\YOUR_RUN\model_4999.pt").Path
& $IsaacPython .\fr5_rl_development\run.py play --num_envs 4 --checkpoint $Checkpoint
& $IsaacPython .\fr5_rl_development\run.py play --num_envs 4 --checkpoint $Checkpoint --steps 650 --headless
& $IsaacPython .\fr5_rl_development\run.py train --num_envs 128 --max_iterations 2000 --headless --checkpoint $Checkpoint
```

Use absolute checkpoint paths, as in the example above. The launcher changes its
working directory to `fr5_rl_development/` to keep outputs together. On resume,
iteration accounting follows the installed RSL-RL backend; inspect the run log.
Earlier custom-arm checkpoints are incompatible with the FR5 task.

### Evaluate completed episodes

```powershell
& $IsaacPython .\fr5_rl_development\evaluate.py --checkpoint $Checkpoint --episodes 32 --num_envs 8 --seed 123 --viz none
```

The report in `fr5_rl_development/evaluation/latest.json` records successes,
success rate, drops, timeouts, the seed, and the checkpoint path. Evaluate multiple
seeds after sufficient training. Reward growth alone does not establish success.
To inspect training metrics with TensorBoard, if available in the environment:

```powershell
& $IsaacPython -m tensorboard.main --logdir .\fr5_rl_development\logs\rsl_rl\fr5_pick_place
```

See [the training guide](fr5_rl_development/TRAINING.md) for task details and checks.

## Task specification

| Property | FR5 task |
| --- | --- |
| Training task ID | `Isaac-FR5-PickPlace-v0` |
| Playback task ID | `Isaac-FR5-PickPlace-Play-v0` |
| Object | 30 mm cube, 30 g |
| Table surface / robot base height | 0.10 m |
| Pick / target region centers | Approximately x = -0.45 m, y = -0.12 / +0.12 m |
| Reset randomization | x ±0.04 m and y ±0.02 m within the separate regions |
| Actions | TCP delta x/y/z (up to 6 mm per step), binary gripper |
| Gripper | Both fingers: open 0 m; close 0.04 m |
| End-effector orientation | Downward, fixed yaw of 90 degrees |
| Control / physics frequency | 50 Hz / 200 Hz |
| Maximum episode duration | 12 seconds |
| Policy input | Simulation state; no camera observations |
| PPO network | Actor and critic MLPs: 256–128–64 |

Success requires the cube to have been lifted at least 45 mm above its resting
height while near the closed gripper, then placed within 20 mm in XY and 8 mm in
height of the target. Cube speed must be below 0.06 m/s, both fingers must be open
below 4 mm, and the TCP must withdraw more than 55 mm. Placement conditions must
hold for 0.30 seconds. Pushing a cube directly to the target does not count.
Coordinates are local to each replicated environment; code quaternions use
`(x, y, z, w)` in this checkout.

## Validation

Repository preparation checks on 21 October 2026 passed: both FR5 task
configurations, both baseline task configurations, all three FR5 logic tests,
and static FR5 asset validation with no unresolved dependencies. Local document
links, 16 Git ignore cases, and the USD/mesh LFS attributes were also checked.
Simulation training and playback were not rerun as part of this documentation update.

Run these checks with the same interpreter:

```powershell
& $IsaacPython -m unittest discover -s .\fr5_rl_development\tests -p "test_*.py"
& $IsaacPython .\fr5_rl_development\tests\smoke_env.py --viz none
& $IsaacPython .\fr5_rl_development\demo.py --viz none --num_envs 4 --seed 42
& $IsaacPython .\fr5_preparation\package\gripper_scene.py --viz none --cycles 2
& $IsaacPython .\fr5_preparation\package\gripper_scene.py --viz none --cycles 2 --gravity
& $IsaacPython .\fr5_preparation\package\check_contact.py --viz none
```

Retained asset records are in `fr5_preparation/package/verification/`. Simulation
checks generate additional records under their respective project directories.

Earlier local documentation dated 21 October 2026 reports three passing logic
tests, a 4/4 scripted demonstration, a 650-step environment check, and a two-update
PPO smoke run. The smoke policy achieved **0/2 successful evaluation episodes**.
Those training records came from the earlier standalone development layout and
are not all included here; see [the archived report](docs/archive/TRAINING.original.vi.md).
Treat these as historical results rather than evidence of a fresh-clone benchmark.

The asset has estimated mass/inertia and drive settings, and self-collision is
disabled. Neither real-robot calibration nor sim-to-real deployment is established.

## Original custom-arm baseline

The root launcher still runs the earlier custom arm:

```powershell
& $IsaacPython .\run.py check
& $IsaacPython .\run.py train --num_envs 8 --max_iterations 2 --headless
& $IsaacPython .\gripper_scene.py --viz kit --cycles 2
& $IsaacPython .\check_gripper.py --viz none
```

Its task is `Isaac-Lift-Cube-Place-Custom-Arm-v0`; logs use
`logs/rsl_rl/franka_lift_place_custom_arm/`. This historical experiment name does
not refer to the FR5. Playback requires a separately obtained compatible checkpoint.
`tests/smoke_play.py` expects a historical checkpoint under the root `logs/` folder;
it cannot run from a clean clone without that file.

The baseline gripper uses `cumotion_arm_gripper_fixed.usda` with limits [-60, 0]
degrees: -60 closes and 0 opens. Earlier checkpoints predate a gripper direction
fix and do not establish successful grasping. See the unchanged
[original README](docs/archive/README.original.vi.md) for the diagnosis and earlier
validation history; its machine-specific paths and checkpoint references are historical.

## Troubleshooting

| Symptom | What to check |
| --- | --- |
| `No module named isaaclab` or `isaaclab_physx` | Use the configured Isaac Lab interpreter and a compatible checkout. |
| `No module named fr5_cfg` | Use the FR5 launcher in this repository and keep `fr5_preparation/package/` beside `fr5_rl_development/`. |
| Asset does not open or dependencies are missing | Run `git lfs pull`; retain USD layers, meshes, and textures together; run `validate_asset.py`. |
| `--checkpoint latest` finds nothing | Train first; checkpoints and local logs are excluded from Git. |
| Playback fails to load a model | Use an FR5 checkpoint with the matching observation/action configuration. |
| GPU memory exhaustion | Reduce `--num_envs` and close other GPU workloads. |
| First launch takes a long time | Initial runtime setup may load extensions/assets and compile GPU kernels. |
| Cache/config write errors | Run from a normal PowerShell session with writable simulator cache/config directories. |
| Older tutorial commands fail | This launcher uses the installed entrypoint API; use the commands above instead of copying old Isaac Lab script paths. |

## Publishing and contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for the first-push workflow, large-file
handling, and verification expectations. Git LFS rules are included; enable LFS
before the first `git add`. Logs, checkpoints, exported policies, and local settings
are excluded. Keep runtime assets and provenance records in the repository.

## Acknowledgements

- **Wang Ziyuan / WangZY233** — original FR5 robot-description source:
  [FR5_Reinforcement-learning](https://github.com/WangZY233/FR5_Reinforcement-learning).
- **Robot Mania** — named maintainer of the retained custom-arm description.
  Original pick-and-place tutorial, identified by the repository maintainer:
  [YouTube tutorial](https://www.youtube.com/watch?v=IGVywY_FI3c).
- **The Isaac Lab Project Developers** — [Isaac Lab](https://github.com/isaac-sim/IsaacLab)
  and existing task foundations; original source headers are preserved.
- **FAIRINO / FAIR-INNOVATION** — nominal arm mass reference from the
  [official FR5 V6 URDF](https://github.com/FAIR-INNOVATION/frcobot_ros2/blob/main/fairino_description/urdf/fairino5_v6.urdf).

Full source and modification notes are in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## License

This repository contains material from multiple sources. Applicable upstream
licenses and notices are preserved in [licenses/](licenses/) and existing file
headers. No blanket license has been assigned to the entire repository or to new
maintainer contributions. See [licensing scope](THIRD_PARTY_NOTICES.md#licensing-scope)
for the remaining provenance and licensing details.


