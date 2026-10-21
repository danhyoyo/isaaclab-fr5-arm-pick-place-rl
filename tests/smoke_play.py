"""Exercise the real playback backend for a finite number of simulation steps.

Only the unbounded playback loop is replaced. Environment creation, checkpoint
loading, export and policy inference use the installed Isaac Lab implementation.
The tutorial checkpoint is copied so its original exports remain untouched.
"""

from pathlib import Path
import runpy
import shutil
import sys
from unittest.mock import patch

import torch
import warp as wp

wp.config.enable_backward = False
ROOT = Path(__file__).resolve().parents[1]


def bounded_playback(step, **kwargs):
    with torch.inference_mode():
        # An episode is 250 steps; this also exercises the reset path.
        for index in range(300):
            step()
            if (index + 1) % 100 == 0:
                print(f"Playback integration check: {index + 1}/300 steps", flush=True)
    print("PASS: bundled checkpoint loaded, exported and played for 300 steps", flush=True)


if __name__ == "__main__":
    checkpoint = ROOT / "logs/rsl_rl/franka_lift_place_custom_arm/2026-05-25_18-19-11/model_3999.pt"
    destination = ROOT / "logs/verification/pretrained/model_3999.pt"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(checkpoint, destination)
    sys.argv = [
        str(ROOT / "run.py"), "play", "--num_envs", "8", "--headless",
        "--checkpoint", str(destination),
    ]
    with patch("isaaclab_rl.entrypoints.backends.play_rsl_rl.run_playback", bounded_playback):
        runpy.run_path(str(ROOT / "run.py"), run_name="__main__")
