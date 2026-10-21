# FR5 training and evaluation

Install the compatible Isaac Lab environment and define `$IsaacPython` using the
[root installation guide](../README.md#installation). All commands here run from
the repository root. The shared robot assets are in `fr5_preparation/package/`.

## Check and demonstrate

```powershell
& $IsaacPython .\fr5_rl_development\run.py check
& $IsaacPython -m unittest discover -s .\fr5_rl_development\tests -p "test_*.py"
& $IsaacPython .\fr5_rl_development\demo.py --viz kit --num_envs 1
```

The demo commands approach, descend, close, lift, transfer, lower, release, and
withdraw through the same four actions used by PPO. The cube is reset only at the
start; it is not attached or teleported during manipulation. A successful demo
validates a scripted controller, not a trained policy.

## Train

```powershell
# Small pipeline check
& $IsaacPython .\fr5_rl_development\run.py train --num_envs 8 --max_iterations 2 --headless --run_name smoke

# Longer training run; adjust environment count for available VRAM
& $IsaacPython .\fr5_rl_development\run.py train --num_envs 128 --max_iterations 5000 --headless --run_name fr5_cube30
```

Runs are stored under `fr5_rl_development/logs/rsl_rl/fr5_pick_place/` regardless of
the calling directory. Use `--viz kit` instead of `--headless` to view training.
These commands use the installed RSL-RL entrypoints, not older tutorial scripts.

## Play, resume, and evaluate

```powershell
& $IsaacPython .\fr5_rl_development\run.py play --num_envs 4 --checkpoint latest

# Replace this example with an existing checkpoint; use an absolute path.
$Checkpoint = (Resolve-Path ".\fr5_rl_development\logs\rsl_rl\fr5_pick_place\YOUR_RUN\model_4999.pt").Path
& $IsaacPython .\fr5_rl_development\run.py play --num_envs 4 --checkpoint $Checkpoint --steps 650 --headless
& $IsaacPython .\fr5_rl_development\run.py train --num_envs 128 --max_iterations 2000 --headless --checkpoint $Checkpoint
& $IsaacPython .\fr5_rl_development\evaluate.py --checkpoint $Checkpoint --episodes 32 --num_envs 8 --seed 123 --viz none
```

`latest` needs an existing FR5 run. Earlier custom-arm checkpoints are incompatible.
Resume iteration accounting follows the installed backend. Evaluation writes
`fr5_rl_development/evaluation/latest.json`, recording the checkpoint, seed,
episode count, successes, drops, timeouts, and individual results.

## Task and policy

- Robot: six-axis FR5 with two sliding gripper fingers; simulation estimates.
- Object: 30 mm cube, 30 g, created in code. Table and robot base height: 0.10 m.
- Pick/goal centers: x = -0.45 m, y = -0.12 / +0.12 m; randomization x ±0.04 m,
  y ±0.02 m. Coordinates are local to each environment.
- Actions: TCP x/y/z target increments up to 6 mm per step and binary gripper
  (`>= 0` opens, `< 0` closes both fingers).
- Finger positions: open = 0 m; closed target = 0.04 m for both `fj1` and `fj2`.
- IK holds downward orientation at yaw 90 degrees; arbitrary object rotation is
  outside the task. Code quaternions use `(x, y, z, w)` in this checkout.
- Control: 50 Hz; physics: 200 Hz; maximum episode length: 12 seconds.
- Policy: simulation-state observations, no cameras; actor/critic MLPs 256–128–64,
  observation normalization, 32 rollout steps per environment.

Reward terms support approach, grasp, lift, transfer, placement, release, withdrawal,
and success, with action-change and joint-velocity penalties. Success requires:

1. Prior lifting at least 45 mm above rest while near the closed gripper.
2. Cube-to-goal XY error below 20 mm and height error below 8 mm.
3. Cube speed below 0.06 m/s and both fingers open below 4 mm.
4. TCP more than 55 mm from the cube, with placement conditions held for 0.30 s.

Pushing the cube to the goal without lifting does not count. Episodes end on
success, leaving the permitted workspace, or timeout. Task-specific arm/wrist
stiffness and damping are 8000/200 and 2000/80 respectively; torque limits remain
150 N·m for j1–j3 and 28 N·m for j4–j6. See source constants/configuration for details.

## Verification and limits

```powershell
& $IsaacPython .\fr5_rl_development\tests\smoke_env.py --viz none
& $IsaacPython .\fr5_rl_development\demo.py --viz none --num_envs 4 --seed 42
```

The smoke check exercises finite observations/rewards, vector origins, and timeout
resets. The demo records results in `fr5_rl_development/verification/`. Re-running
checks can update verification JSON. Keep deliberate benchmark records separate
from temporary experiments.

[The archived local training report](../docs/archive/TRAINING.original.vi.md)
records a 4/4 scripted demo and a two-update smoke policy with 0/2 evaluation
successes on 21 October 2026. Not all original training artifacts are included;
these historical results do not establish learned-policy performance for a new run.
The asset is not calibrated to real hardware and self-collision remains disabled.
