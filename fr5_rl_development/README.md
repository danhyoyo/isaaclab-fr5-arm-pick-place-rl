# FR5 reinforcement-learning workflow

This directory contains the FR5 task and PPO utilities. The robot configuration
and assets are shared from `../fr5_preparation/package/`; `run.py` resolves that
location automatically. Run commands from the repository root as documented in
the [main README](../README.md).

The maintainer's contribution is the adaptation of the pick-and-place workflow
to the FAIRINO FR5. Robot assets originate from Wang Ziyuan's
[FR5_Reinforcement-learning](https://github.com/WangZY233/FR5_Reinforcement-learning).
See [source notices](../THIRD_PARTY_NOTICES.md) for the complete attribution.

```powershell
$IsaacPython = "D:\IsaacLab\.venv\Scripts\python.exe"
& $IsaacPython .\fr5_rl_development\run.py check
& $IsaacPython .\fr5_rl_development\demo.py --viz kit --num_envs 1
& $IsaacPython .\fr5_rl_development\run.py train --num_envs 128 --max_iterations 5000 --headless --run_name fr5_cube30
& $IsaacPython .\fr5_rl_development\run.py play --num_envs 4 --checkpoint latest
```

See [TRAINING.md](TRAINING.md) for the task specification, validation, evaluation,
and runtime output paths. The demo uses scripted waypoints; learned-policy success
must be measured separately. No trained checkpoint is included.

Previous Vietnamese documentation is preserved in
[docs/archive/FR5.original.vi.md](../docs/archive/FR5.original.vi.md).
