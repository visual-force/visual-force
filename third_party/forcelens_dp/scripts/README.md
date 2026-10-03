# Task launchers

The task launchers are thin wrappers around `policy_server.py`. Run them from
this directory with the project environment active. They require external
policy and force-estimator checkpoints; no model weights or datasets are
shipped in VisualForce.

| Task | Force-aware inference | Monitor-only baseline |
| --- | --- | --- |
| Berry | `scripts/berry tts` | `scripts/berry baseline` |
| Coke | `scripts/coke tts` | `scripts/coke baseline` |
| Reorientation | `scripts/flip tts` | `scripts/flip raw_dp` |
| Plug insertion | `scripts/plug tts` | `scripts/plug baseline` |

Set `VISUALFORCE_CKPT` for the external image force estimator and `CKPT_PATH`
for the task policy. See each task launcher's `--help` output for optional
overrides. The canonical controller contracts live in
`inference_profiles.py`.

For the paper-compatible Coke action-force policy, use
`scripts/coke force-output ...`; it trains an explicit 9D action vector with
the eighth robot action followed by the magnitude of `Fz`.
