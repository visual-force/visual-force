import argparse

from inference_profiles import apply_profile_defaults, get_profile


def _args(profile):
    return argparse.Namespace(
        tts_experiment_profile=profile,
        tts_desired_force=None,
        tts_force_target_mode="absolute",
        tts_selection_scope="full",
        tts_sampling_candidates=8,
        tts_sampling_score_steps=4,
        tts_activation_force=0.0,
        tts_policy_force_aggregation="last",
        tts_policy_force_dynamic_target_rise=None,
        tts_policy_force_output=False,
        tts_close_positive=False,
        tts_close_negative=False,
        tts_gentle_gripper_control=False,
        tts_add_gripper_fallback_candidates=False,
        tts_disable_policy_release=False,
    )


def test_final_profiles_encode_distinct_controller_contracts():
    coke = get_profile("coke")
    berry = get_profile("berry")
    reorientation = get_profile("flip")
    plug = get_profile("plug")

    assert (coke.target_mode, coke.selection_scope, coke.gentle_gripper_control) == (
        "baseline_delta",
        "gripper",
        False,
    )
    assert (berry.activation_force, berry.policy_force_aggregation) == (0.5, "max")
    assert reorientation.selection_scope == "full"
    assert reorientation.policy_force_dynamic_target_rise == 0.2
    assert (plug.target_mode, plug.policy_force_dynamic_target_rise) == ("absolute", 1.23)


def test_profile_defaults_preserve_explicit_cli_overrides():
    args = _args("coke")
    apply_profile_defaults(args, ["--tts-experiment-profile", "coke"])
    assert args.tts_desired_force == 3.0
    assert args.tts_selection_scope == "gripper"
    assert args.tts_policy_force_output is True

    args = _args("coke")
    apply_profile_defaults(
        args,
        [
            "--tts-experiment-profile",
            "coke",
            "--tts-desired-force",
            "9",
            "--tts-selection-scope",
            "full",
        ],
    )
    assert args.tts_desired_force is None
    assert args.tts_selection_scope == "full"
