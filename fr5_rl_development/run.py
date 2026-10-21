"""Independent FR5 task launcher using the existing Isaac Lab Python environment."""
import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
ASSET_ROOT = ROOT.parent / "fr5_preparation" / "package"


def register_tasks():
    # The repository keeps the prepared robot assets separate from the RL code.
    if str(ASSET_ROOT) not in sys.path:
        sys.path.insert(0, str(ASSET_ROOT))
    sys.path.insert(0, str(ROOT)) if str(ROOT) not in sys.path else None
    import fr5_pick_place
    return fr5_pick_place


def resolve_play_checkpoint(arguments, tasks):
    """The backend manifest selector uses the training task id, even for a separate play registration."""
    selector = argparse.ArgumentParser(add_help=False)
    selector.add_argument("--checkpoint")
    selector.add_argument("--experiment_name", default="fr5_pick_place")
    selector.add_argument("--agent", default="rsl_rl_cfg_entry_point")
    selection, _ = selector.parse_known_args(arguments)
    if selection.checkpoint != "latest":
        return arguments
    from isaaclab_rl.entrypoints.common import resolve_checkpoint_selector
    checkpoint = resolve_checkpoint_selector(
        str(ROOT / "logs/rsl_rl" / selection.experiment_name), "latest", library="rsl_rl",
        task=tasks.TASK, checkpoint_pattern=r"model_.*\.pt", metadata={"agent": selection.agent})
    result = list(arguments)
    for i, value in enumerate(result):
        if value == "--checkpoint":
            result[i + 1] = str(checkpoint)
        elif value == "--checkpoint=latest":
            result[i] = "--checkpoint=" + str(checkpoint)
    print(f"Resolved latest FR5 checkpoint: {checkpoint}", flush=True)
    return result


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["check", "train", "play"])
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--steps", type=int, help="Optional bounded playback (play mode only).")
    args, remaining = parser.parse_known_args()
    if args.steps is not None and (args.mode != "play" or args.steps < 1):
        parser.error("--steps must be positive and is available only in play mode")
    import warp as wp
    wp.config.enable_backward = False
    tasks = register_tasks()
    os.chdir(ROOT)
    if args.mode == "check":
        if remaining:
            parser.error("check does not accept additional arguments")
        from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
        from fr5_cfg import ROBOT_USD
        from fr5_pick_place.constants import CUBE_SIZE
        assert ROBOT_USD.is_file(), ROBOT_USD
        for name in [tasks.TASK, tasks.PLAY_TASK]:
            cfg = load_cfg_from_registry(name, "env_cfg_entry_point")
            cfg.validate()
            agent = load_cfg_from_registry(name, "rsl_rl_cfg_entry_point")
            assert agent.actor.hidden_dims and agent.critic.hidden_dims
            print(f"OK: {name}; cube={CUBE_SIZE * 1000:.0f} mm; policy=TCP xyz + gripper; asset={ROBOT_USD}")
        return 0
    has_viz = any(s in ["--viz", "--visualizer"] or s.startswith(("--viz=", "--visualizer=")) for s in remaining)
    if args.headless and has_viz:
        parser.error("Use --headless or --viz, not both")
    defaults = ["--viz", "none"] if args.headless else ([] if has_viz else ["--viz", "kit"])
    if not any(s == "--task" or s.startswith("--task=") for s in remaining):
        defaults += ["--task", tasks.PLAY_TASK if args.mode == "play" else tasks.TASK]
    defaults += ["--rl_library", "rsl_rl"]
    from isaaclab_rl.entrypoints import run_train_cli, run_play_cli
    if args.mode == "play":
        remaining = resolve_play_checkpoint(remaining, tasks)
    if args.mode == "play" and args.steps:
        from unittest.mock import patch
        import torch
        def bounded(step, **kwargs):
            with torch.inference_mode():
                for _ in range(args.steps):
                    step()
            print(f"PASS: policy loaded and played {args.steps} steps", flush=True)
        with patch("isaaclab_rl.entrypoints.backends.play_rsl_rl.run_playback", bounded):
            return run_play_cli(defaults + remaining)
    return (run_train_cli if args.mode == "train" else run_play_cli)(defaults + remaining)


if __name__ == "__main__":
    raise SystemExit(main())
