"""Run this external task with the Python environment that contains Isaac Lab.

Examples:
    D:\\IsaacLab\\.venv\\Scripts\\python.exe run.py check
    D:\\IsaacLab\\.venv\\Scripts\\python.exe run.py train --num_envs 8 --max_iterations 2 --headless
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
TASK = "Isaac-Lift-Cube-Place-Custom-Arm-v0"
PLAY_TASK = "Isaac-Lift-Cube-Place-Custom-Arm-Play-v0"


def register_tasks() -> None:
    """Load this task package without installing it into the Isaac Lab venv."""
    if "pick_and_place.custom_arm" in sys.modules:
        return
    spec = importlib.util.spec_from_file_location(
        "pick_and_place", PROJECT_ROOT / "__init__.py",
        submodule_search_locations=[str(PROJECT_ROOT)],
    )
    package = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = package
    spec.loader.exec_module(package)
    importlib.import_module("pick_and_place.custom_arm")


def main() -> int:
    # Isaac Lab prints Unicode status panels, including when output is redirected.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("check", "train", "play"))
    parser.add_argument("--headless", action="store_true", help="Run without a visualizer.")
    args, remaining = parser.parse_known_args()

    import warp as wp
    wp.config.enable_backward = False
    register_tasks()

    # Isaac Lab stores logs relative to cwd. Keep them with this project.
    os.chdir(PROJECT_ROOT)

    if args.mode == "check":
        if remaining:
            parser.error("check does not accept extra arguments")
        from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

        for task in (TASK, PLAY_TASK):
            cfg = load_cfg_from_registry(task, "env_cfg_entry_point")
            cfg.validate()
            agent = load_cfg_from_registry(task, "rsl_rl_cfg_entry_point")
            # Runner configs contain intentionally missing deprecated fields;
            # RSL-RL resolves those when constructing the runner.
            if not agent.actor.hidden_dims or not agent.critic.hidden_dims:
                raise ValueError(f"Missing actor/critic hidden dimensions for {task}")
            for asset in (cfg.scene.robot, cfg.scene.object):
                path = Path(asset.spawn.usd_path)
                if not path.is_file():
                    raise FileNotFoundError(path)
            print(f"OK: {task}; physics={type(cfg.sim.physics).__name__}")
        return 0

    from isaaclab_rl.entrypoints import run_train_cli, run_play_cli

    has_visualizer = any(
        arg in ("--viz", "--visualizer") or arg.startswith(("--viz=", "--visualizer="))
        for arg in remaining
    )
    if args.headless and has_visualizer:
        parser.error("Use either --headless or --viz, not both")
    defaults = ["--viz", "none"] if args.headless else []
    if args.mode == "play" and not args.headless and not has_visualizer:
        defaults += ["--viz", "kit"]
    if not any(arg == "--task" or arg.startswith("--task=") for arg in remaining):
        defaults += ["--task", PLAY_TASK if args.mode == "play" else TASK]
    if not any(arg == "--rl_library" or arg.startswith("--rl_library=") for arg in remaining):
        defaults += ["--rl_library", "rsl_rl"]
    run_cli = run_train_cli if args.mode == "train" else run_play_cli
    return run_cli(defaults + remaining)


if __name__ == "__main__":
    raise SystemExit(main())
