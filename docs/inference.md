# Inference

VisualForce adds force-aware candidate selection to a frozen diffusion policy.
The robot-policy implementation lives in `third_party/forcelens_dp` and is
served by `policy_server.py`. Datasets and all model weights remain external.

## Required external artifacts

| Artifact | Argument | Purpose |
| --- | --- | --- |
| Diffusion policy | `CKPT_PATH` / `--ckpt-path` | Produces robot action chunks |
| Force estimator | `VISUALFORCE_CKPT` / `--tts-visualforce-ckpt` | Estimates current force |
| SAM2 weights | `--tts-sam2-ckpt` | Produces the gripper mask |
| Force critic, optional | `CRITIC_CKPT` | Legacy Coke critic mode only |

No default checkpoint path is stored in the repository. Supply paths explicitly
for reproducible runs:

```bash
cd /path/to/VisualForce/third_party/forcelens_dp
VISUALFORCE_CKPT=/path/to/force-estimator.pt \
CKPT_PATH=/path/to/policy.ckpt \
scripts/coke tts
```

## Canonical experiment profiles

The profile registry is `inference_profiles.py`; `policy_server.py` applies it
only when a command-line override is absent.

| Profile | Force target | Action ownership | Special behavior |
| --- | --- | --- | --- |
| `coke` | baseline-relative, 3.0 N | selected gripper trajectory | no direct gripper controller |
| `berry` | baseline-relative, 5.0 N | selected gripper trajectory | gentle gripper feedback, hold/open fallbacks |
| `reorientation` / `flip` | baseline-relative, 4.0 N | complete chunk | activates at 3.0 N; dynamic rise 0.2 N |
| `plug_insertion` / `plug` | absolute, 12.0 N cap | complete chunk | dynamic rise 1.23 N |

Launchers:

```bash
scripts/berry tts --no-interactive
scripts/coke tts
scripts/flip tts
scripts/plug tts
```

Use `baseline`, `raw_dp`, or `dp` for monitor-only comparisons. TTS reranks
learned policy chunks; it is not a robot-side safety controller or axial
admittance controller. Keep the robot-side watchdog and an attended stop path
active for hardware trials.

## Observation and masking

The standard force-aware path uses the wrist RGB observation, automatic SAM2
masking, and the force estimator's saved input configuration. The task launcher
passes the frame key and segmentation settings to the policy server. Do not
mix a policy checkpoint with incompatible action representation or observation
keys.

## Rollouts

Set `ROLLOUT_DIR` to an ignored local directory when recording diagnostics.
Typical files include:

```text
force_log.csv          force estimates and selected actions
candidate_scores.csv   optional per-candidate scores
original_h264.mp4      wrist-camera frames
masked_original_h264.mp4
edge_h264.mp4
side_view_h264.mp4     optional side-camera frames
```

Rollouts, logs, videos, and reports must remain outside version control.

## Hardware boundary

The launchers can perform GPU preflight and dry-run validation without hardware
motion. Do not start robot motion, homing, inference, or data collection unless
an operator has explicitly requested it and is supervising the run.
