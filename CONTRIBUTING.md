# Contributing and publishing

## First push to GitHub

From the repository root, install Git LFS and initialize the local repository:

```powershell
git init -b main
git lfs install --local
git add .
git status --short
git lfs ls-files
```

Review the staged files. Include source, documentation, USD layers, URDF, meshes,
textures, license texts, parameters, and source manifests. Logs, policy checkpoints,
cache directories, and private environment settings should be excluded.
The FR5 base USD layer is approximately 95.7 MiB; keep the supplied LFS rules.
Git LFS downloads must also be available to anyone cloning the project.

After reviewing attribution in `THIRD_PARTY_NOTICES.md`, create an empty GitHub
repository and replace the example remote URL below with its actual URL:

```powershell
git commit -m "Add FR5 pick-and-place adaptation for Isaac Lab"
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
git push -u origin main
```

These commands are instructions for a new repository. If Git is already initialized
or a remote exists, inspect its current branch/remotes before changing them.
The documentation preparation itself does not publish anything.

## Changes and verification

Describe the problem, changed behavior, environment versions, and relevant checks.
Preserve upstream source headers and credit authors when importing code or assets.
Record robot physics changes in `fr5_preparation/package/parameters.json` and rerun
asset validation. Do not regenerate the source manifest from modified output;
it records hashes of original inputs.

For task-logic changes, run the FR5 logic tests and the configuration check. For
simulation/control changes, also run the environment smoke check and scripted
demo. For asset/gripper changes, run static, motion, and contact checks. Commands
are listed in the root README. Report seed, environment count, and checkpoint for
policy evaluations. Keep scripted-controller results separate from learned-policy
results.

Avoid committing generated training output. Share trained checkpoints as separately
identified release artifacts with the compatible task configuration and evaluation
results. The repository's ignore rules exclude checkpoints and exported policies.
