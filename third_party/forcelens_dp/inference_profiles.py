"""Canonical inference contracts for the four completed VisualForce tasks.

The experiment launchers used to carry these values independently.  Keeping
the contract here makes the policy server the single source of truth while
still allowing a launcher or operator to override an individual option.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class InferenceProfile:
    name: str
    target_mode: str
    desired_force: float
    selection_scope: str
    candidates: int = 32
    score_steps: int = 4
    activation_force: float = 0.0
    policy_force_aggregation: str = "last"
    policy_force_dynamic_target_rise: Optional[float] = None
    policy_force_output: bool = True
    close_positive: bool = True
    gentle_gripper_control: bool = False
    add_gripper_fallback_candidates: bool = False
    policy_release_enabled: bool = True


PROFILES = {
    # Final Coke controller: learned-force gripper-only reranking, no direct
    # force controller and no handcrafted gripper trajectory.
    "coke": InferenceProfile(
        "coke", "baseline_delta", 3.0, "gripper"
    ),
    # Final Berry controller: learned-force reranking plus the gentle direct
    # gripper loop and its hold/open fallback candidates.
    "berry": InferenceProfile(
        "berry",
        "baseline_delta",
        5.0,
        "gripper",
        activation_force=0.5,
        policy_force_aggregation="max",
        gentle_gripper_control=True,
        add_gripper_fallback_candidates=True,
    ),
    # Reorientation's final policy owns the complete action chunk.
    "reorientation": InferenceProfile(
        "reorientation",
        "baseline_delta",
        4.0,
        "full",
        activation_force=3.0,
        policy_force_dynamic_target_rise=0.2,
    ),
    # Plug insertion uses an absolute terminal cap, despite its dynamic
    # candidate target being based on the sampled policy force.
    "plug_insertion": InferenceProfile(
        "plug_insertion",
        "absolute",
        12.0,
        "full",
        policy_force_dynamic_target_rise=1.23,
    ),
}

ALIASES = {"flip": "reorientation", "plug": "plug_insertion"}


def get_profile(name: str) -> InferenceProfile:
    canonical = ALIASES.get(name, name)
    try:
        return PROFILES[canonical]
    except KeyError as exc:
        choices = ", ".join(sorted(PROFILES))
        raise ValueError(f"unknown inference profile {name!r}; choose {choices}") from exc


def apply_profile_defaults(args, argv):
    """Apply profile values only when the corresponding CLI flag was omitted.

    This preserves existing operator overrides while removing duplicated task
    defaults from the shell launchers.
    """

    profile_name = getattr(args, "tts_experiment_profile", None)
    if profile_name is None:
        return args
    profile = get_profile(profile_name)

    def omitted(flag):
        return flag not in argv

    values = {
        "tts_desired_force": profile.desired_force,
        "tts_force_target_mode": profile.target_mode,
        "tts_selection_scope": profile.selection_scope,
        "tts_sampling_candidates": profile.candidates,
        "tts_sampling_score_steps": profile.score_steps,
        "tts_activation_force": profile.activation_force,
        "tts_policy_force_aggregation": profile.policy_force_aggregation,
        "tts_policy_force_dynamic_target_rise": profile.policy_force_dynamic_target_rise,
    }
    for attr, value in values.items():
        flag = "--" + attr.replace("_", "-")
        if value is not None and omitted(flag):
            setattr(args, attr, value)

    if profile.policy_force_output and omitted("--tts-policy-force-output"):
        args.tts_policy_force_output = True
    if profile.close_positive and omitted("--tts-close-positive") and omitted("--tts-close-negative"):
        args.tts_close_positive = True
    if profile.gentle_gripper_control and omitted("--tts-gentle-gripper-control"):
        args.tts_gentle_gripper_control = True
    if profile.add_gripper_fallback_candidates and omitted("--tts-add-gripper-fallback-candidates"):
        args.tts_add_gripper_fallback_candidates = True
    if not profile.policy_release_enabled and omitted("--tts-disable-policy-release"):
        args.tts_disable_policy_release = True
    return args
