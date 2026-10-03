# Diffusion-policy runtime

This directory contains the retained diffusion-policy runtime used by
VisualForce. It is not a data or checkpoint distribution.

Run commands from this directory with the project environment active. The
formal task contracts are defined in `inference_profiles.py` and documented in
the repository root README.

```bash
VISUALFORCE_CKPT=/path/to/force-estimator.pt \
CKPT_PATH=/path/to/policy.ckpt \
scripts/berry tts --no-interactive
```

Available final task launchers:

```text
scripts/berry  berry grasp profile
scripts/coke   Coke gripper-only policy sampling
scripts/flip   reorientation full-chunk selection
scripts/plug   plug-insertion full-chunk selection
```

`baseline`, `raw_dp`, and `dp` commands monitor force without changing policy
actions. TTS commands select learned policy chunks and must not be treated as
hardware safety controllers. All policy checkpoints, force-estimator weights,
SAM2 weights, datasets, logs, and rollout videos are external artifacts.

For the paper-compatible Coke action-force policy, use
`scripts/coke force-output {build|label|validate|train|all|dry-run}`. It trains
an explicit 9D action vector with eight robot actions followed by `|Fz|`.

See:

- [`../../README.md`](../../README.md) for the repository contract;
- [`../../docs/inference.md`](../../docs/inference.md) for runtime behavior;
- [`scripts/README.md`](scripts/README.md) for launcher names.
