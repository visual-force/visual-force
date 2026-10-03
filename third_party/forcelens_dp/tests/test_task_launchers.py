import os
import subprocess
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


class TestTaskLaunchers(unittest.TestCase):
    def launcher_command(self, task, action, *extra):
        env = os.environ.copy()
        env['PYTHON_BIN'] = '/bin/echo'
        for name in (
            'CKPT_PATH',
            'CKPT_PRESET',
            'DESIRED_FORCE',
            'ACTIVATION_FORCE',
            'POLICY_FORCE_RISE',
            'FORCE_AGGREGATION',
            'DEVICE',
            'SAMPLING_CANDIDATES',
            'POLICY_ACTION_STEPS',
            'N_ACTION_STEPS',
            'ACTION_REFILL_STEPS',
            'TTS_CONTACT_FORCE_DELTA',
            'TTS_FORCE_FILTER_WINDOW',
            'TTS_FORCE_BASELINE_SAMPLES',
            'TTS_GRIPPER_SAFETY_MIN_POSITION',
            'TTS_GRIPPER_MAX_POSITION',
            'TTS_GENTLE_DEADBAND',
            'TTS_GENTLE_STOP_MARGIN',
            'LOAD_DEADBAND_N',
            'CLOSURE_PER_LOAD_N',
            'FORCE_LIMIT_RISE_N',
            'MAX_CLOSURE',
            'MAX_CLOSURE_DELTA',
            'PAYLOAD_TORQUE_SIGN',
        ):
            env.pop(name, None)
        result = subprocess.run(
            [str(REPO_ROOT / 'scripts' / task), action, *extra],
            cwd=REPO_ROOT,
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.split()

    def argument(self, command, name):
        index = command.index(name)
        return command[index + 1]

    def test_cup_server_uses_visualforce_without_a_robot_policy(self):
        monitor = self.launcher_command('cup', 'monitor')
        adaptive = self.launcher_command('cup', 'adaptive')

        for command in (monitor, adaptive):
            self.assertIn('adaptive_grip_server.py', command)
            self.assertIn('--visualforce-ckpt', command)
            self.assertIn('--sam2-ckpt', command)
            self.assertNotIn('--ckpt-path', command)
            self.assertNotIn('policy_server.py', command)
            self.assertEqual(self.argument(command, '--max-closure'), '0.55')
            self.assertEqual(
                self.argument(command, '--max-closure-delta'),
                '0.03',
            )
            self.assertEqual(self.argument(command, '--closure-per-load-n'), '0.025')
            self.assertEqual(self.argument(command, '--force-limit-rise-n'), '3.0')

        self.assertIn('--monitor-only', monitor)
        self.assertNotIn('--monitor-only', adaptive)

    def test_cup_pose_commands_only_forward_relevant_options(self):
        capture = self.launcher_command('cup', 'pose', 'capture')
        replace = self.launcher_command('cup', 'pose', 'replace')
        goto = self.launcher_command('cup', 'pose', 'goto')
        show = self.launcher_command('cup', 'pose', 'show')

        self.assertIn('--capture-samples', capture)
        self.assertNotIn('--max-translation-step-m', capture)
        self.assertIn('--overwrite', replace)
        self.assertIn('--max-translation-step-m', goto)
        self.assertNotIn('--capture-samples', goto)
        self.assertNotIn('--port', show)
        self.assertNotIn('--stationary-velocity-limit', show)

    def test_baseline_and_tts_share_latest_task_checkpoint(self):
        for task, tts_extra in (
            ('berry', ('--no-interactive',)),
            ('flip', ()),
            ('coke', ()),
        ):
            with self.subTest(task=task):
                tts = self.launcher_command(task, 'tts', *tts_extra)
                baseline = self.launcher_command(task, 'baseline')
                tts_checkpoint = self.argument(tts, '--ckpt-path')
                baseline_checkpoint = self.argument(baseline, '--ckpt-path')
                self.assertEqual(tts_checkpoint, baseline_checkpoint)
                self.assertEqual(Path(tts_checkpoint).name, 'latest.ckpt')
                self.assertEqual(
                    self.argument(tts, '--tts-steering-mode'),
                    'sample',
                )
                self.assertEqual(
                    self.argument(baseline, '--tts-steering-mode'),
                    'monitor',
                )

    def test_flip_force_output_dry_run_trains_ninth_pseudo_force_channel(self):
        command = self.launcher_command('flip', 'force-output', 'dry-run')

        self.assertIn('train.py', command)
        self.assertIn('task.shape_meta.action.shape=[9]', command)
        self.assertIn('task.dataset.append_force_to_action=true', command)
        self.assertIn('task.dataset.force_key=Fz', command)
        self.assertIn('task.dataset.force_mode=magnitude', command)
        self.assertIn('task.dataset.relative_position_action=true', command)
        self.assertIn('task.dataset.relative_position_action_mode=obs_anchor', command)
        self.assertIn(
            'task.dataset_path=data/flip/flip_obj_4_stage2_force_output',
            command,
        )

    def test_baselines_do_not_enable_action_changing_force_flags(self):
        changing_flags = {
            '--tts-gentle-gripper-control',
            '--tts-absolute-gripper-safety',
            '--tts-policy-force-output',
            '--tts-auto-policy-force-output',
            '--tts-add-gripper-fallback-candidates',
            '--tts-delta-force-critic-ckpt',
        }
        for task in ('berry', 'flip', 'coke', 'plug'):
            with self.subTest(task=task):
                baseline = set(self.launcher_command(task, 'baseline'))
                self.assertTrue(changing_flags.isdisjoint(baseline))

    def test_berry_uses_gentle_real_world_defaults(self):
        command = self.launcher_command('berry', 'tts', '--no-interactive')
        self.assertIn('--tts-gentle-gripper-control', command)
        self.assertIn('--tts-policy-force-output', command)
        self.assertEqual(
            self.argument(command, '--tts-selection-scope'),
            'gripper',
        )
        self.assertEqual(self.argument(command, '--tts-desired-force'), '5.0')
        self.assertEqual(
            self.argument(command, '--tts-force-target-mode'),
            'baseline_delta',
        )
        self.assertEqual(
            self.argument(command, '--tts-force-filter-window'),
            '3',
        )
        self.assertEqual(
            self.argument(command, '--tts-contact-force-delta'),
            '5.0',
        )
        self.assertEqual(
            self.argument(command, '--tts-gripper-max-position'),
            '0.82',
        )
        self.assertEqual(
            self.argument(command, '--tts-gripper-max-lead'),
            '0.25',
        )

    def test_enabled_tts_profiles_share_force_and_queue_contract(self):
        for task, extra, target_mode in (
            ('berry', ('--no-interactive',), 'baseline_delta'),
            ('flip', (), 'baseline_delta'),
            ('coke', (), 'baseline_delta'),
            ('plug', (), 'absolute'),
        ):
            with self.subTest(task=task):
                command = self.launcher_command(task, 'tts', *extra)
                self.assertEqual(
                    self.argument(command, '--tts-force-target-mode'),
                    target_mode,
                )
                self.assertEqual(
                    self.argument(command, '--tts-sampling-candidates'),
                    '32',
                )
                self.assertEqual(
                    self.argument(command, '--n-action-steps'),
                    '8',
                )
                self.assertEqual(
                    self.argument(command, '--action-refill-steps'),
                    '2',
                )

    def test_task_profiles_have_explicit_action_ownership(self):
        berry = self.launcher_command('berry', 'tts', '--no-interactive')
        coke = self.launcher_command('coke', 'tts')
        flip = self.launcher_command('flip', 'tts')
        plug = self.launcher_command('plug', 'tts')

        self.assertIn('--tts-gentle-gripper-control', berry)
        self.assertEqual(
            self.argument(berry, '--tts-selection-scope'),
            'gripper',
        )
        self.assertNotIn('--tts-gentle-gripper-control', coke)
        self.assertIn('--tts-policy-force-output', coke)
        self.assertEqual(
            self.argument(coke, '--tts-selection-scope'),
            'gripper',
        )
        self.assertIn('--tts-policy-force-output', flip)
        self.assertNotIn('--tts-delta-force-critic-ckpt', flip)
        self.assertNotIn('--tts-gentle-gripper-control', flip)
        self.assertEqual(self.argument(flip, '--tts-selection-scope'), 'full')
        self.assertIn('--tts-policy-force-output', plug)
        self.assertNotIn('--tts-gentle-gripper-control', plug)
        self.assertEqual(self.argument(plug, '--tts-selection-scope'), 'full')

    def test_flip_defaults_to_completed_force_output_checkpoint(self):
        tts = self.launcher_command('flip', 'tts')
        raw_dp = self.launcher_command('flip', 'raw_dp')

        for command in (tts, raw_dp):
            self.assertIn(
                '19.59.26_train_diffusion_unet_flip_obj4_stage2_force_output',
                self.argument(command, '--ckpt-path'),
            )
        self.assertIn('--tts-policy-force-output', tts)
        self.assertEqual(
            self.argument(tts, '--tts-policy-force-dynamic-target-rise'),
            '0.2',
        )
        self.assertEqual(self.argument(tts, '--tts-desired-force'), '4.0')
        self.assertEqual(self.argument(tts, '--tts-activation-force'), '3.0')
        self.assertEqual(self.argument(raw_dp, '--tts-steering-mode'), 'monitor')

    def test_flip_tts_forces_full_arm_selection(self):
        old_scope = os.environ.get('TTS_SELECTION_SCOPE')
        os.environ['TTS_SELECTION_SCOPE'] = 'gripper'
        try:
            command = self.launcher_command('flip', 'tts')
        finally:
            if old_scope is None:
                os.environ.pop('TTS_SELECTION_SCOPE', None)
            else:
                os.environ['TTS_SELECTION_SCOPE'] = old_scope

        self.assertEqual(self.argument(command, '--tts-selection-scope'), 'full')

    def test_coke_legacy_proxy_is_explicitly_absolute(self):
        command = self.launcher_command('coke', 'proxy', '2.5')
        self.assertEqual(
            self.argument(command, '--tts-force-target-mode'),
            'absolute',
        )
        self.assertNotIn('--tts-gentle-gripper-control', command)

    def test_coke_8d_critic_tts_only_selects_gripper(self):
        command = self.launcher_command('coke', 'tts-critic', '4')
        self.assertEqual(self.argument(command, '--tts-desired-force'), '4')
        self.assertEqual(
            self.argument(command, '--tts-force-target-mode'),
            'baseline_delta',
        )
        self.assertEqual(
            self.argument(command, '--tts-selection-scope'),
            'gripper',
        )
        self.assertIn('--tts-delta-force-critic-ckpt', command)
        self.assertNotIn('--tts-policy-force-output', command)
        self.assertNotIn('--tts-add-gripper-fallback-candidates', command)
        self.assertNotIn('--tts-gentle-gripper-control', command)
        self.assertEqual(self.argument(command, '--policy-action-steps'), '16')
        self.assertEqual(self.argument(command, '--n-action-steps'), '8')

    def test_coke_uses_gripper_only_policy_sampling(self):
        command = self.launcher_command('coke', 'tts')
        self.assertEqual(self.argument(command, '--tts-desired-force'), '3.0')
        self.assertEqual(
            self.argument(command, '--tts-force-target-mode'),
            'baseline_delta',
        )
        self.assertEqual(
            self.argument(command, '--tts-selection-scope'),
            'gripper',
        )
        forbidden = {
            '--tts-gentle-gripper-control',
            '--tts-add-gripper-fallback-candidates',
            '--tts-policy-gripper-approach',
            '--tts-contact-force-delta',
            '--tts-policy-release-contact-delta',
            '--tts-gripper-close-step',
            '--tts-gripper-max-position',
            '--gripper-command-min',
            '--gripper-command-max',
        }
        self.assertTrue(forbidden.isdisjoint(command))

    def test_plug_baseline_and_tts_share_selected_force_output_checkpoint(self):
        tts = self.launcher_command('plug', 'tts')
        baseline = self.launcher_command('plug', 'baseline')
        tts_checkpoint = self.argument(tts, '--ckpt-path')
        baseline_checkpoint = self.argument(baseline, '--ckpt-path')
        self.assertEqual(tts_checkpoint, baseline_checkpoint)
        self.assertEqual(
            Path(tts_checkpoint).name,
            'latest.ckpt',
        )
        self.assertEqual(self.argument(tts, '--tts-desired-force'), '12.0')
        self.assertEqual(
            self.argument(tts, '--tts-policy-force-dynamic-target-rise'),
            '1.23',
        )
        self.assertEqual(
            self.argument(tts, '--tts-force-target-mode'),
            'absolute',
        )
        self.assertEqual(self.argument(baseline, '--tts-desired-force'), '12.0')
        self.assertEqual(
            self.argument(baseline, '--tts-force-target-mode'),
            'absolute',
        )
        self.assertEqual(
            self.argument(tts, '--tts-steering-mode'),
            'sample',
        )
        self.assertEqual(
            self.argument(baseline, '--tts-steering-mode'),
            'monitor',
        )


if __name__ == '__main__':
    unittest.main()
