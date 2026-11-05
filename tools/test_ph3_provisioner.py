#!/usr/bin/env python3
"""Exercise optional P-H3 Image manifests through the real runtime, offline."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from validate_models import runtime_dir

REPO = Path(__file__).resolve().parents[1]
BASE = 'https://huggingface.co/PrunaAI/P-H3-Image-Comfy/resolve/main'
UPSAMPLER = 'p-h3-image_prompt_upsampler_int8_convrot.safetensors'


def expected():
    return {
        'p-h3-image_int8_convrot.safetensors': 'diffusion_models',
        'p-h3-image_text_encoder_int8_convrot.safetensors': 'text_encoders',
        'p-h3-image_vae_int8_convrot.safetensors': 'vae',
        UPSAMPLER: 'text_encoders',
    }


class PH3Provisioning(unittest.TestCase):
    def provision(self, flag=None, minimax='false', viggle='false'):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            models = root / 'models'
            manifest = root / 'manifest.tsv'
            env = dict(os.environ)
            for key in ('download_p_h3_image', 'p_h3_image_quant', 'minimax_quant'):
                env.pop(key, None)
            env.update(download_minimax_h3=minimax, download_viggle_animate=viggle)
            if flag is not None:
                env['download_p_h3_image'] = flag
            result = subprocess.run([
                sys.executable, str(runtime_dir() / 'src/provisioner.py'),
                '--template', str(REPO / 'template.json'),
                '--registry', str(REPO / 'src/models_registry.json'),
                '--workflows-src', str(REPO / 'workflows'),
                '--workflows-dst', str(root / 'workflows'),
                '--models-root', str(models), '--manifest', str(manifest),
            ], env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertNotIn('[provisioner] error:', result.stdout)
            entries = [line.split('\t')[:2] for line in manifest.read_text().splitlines() if line]
            actual = {Path(dest).name: (url, str(Path(dest).relative_to(models)))
                      for url, dest in entries}
            self.assertEqual(len(actual), len(entries))
            return actual

    def assert_bundle(self, entries):
        actual = {name: value for name, value in entries.items() if name.startswith('p-h3-image_')}
        self.assertEqual(actual, {
            name: (f'{BASE}/{category}/{name}', f'{category}/{name}')
            for name, category in expected().items()
        })

    def test_configuration(self):
        template = json.loads((REPO / 'template.json').read_text())
        registry = json.loads((REPO / 'src/models_registry.json').read_text())
        self.assertEqual(template['flags']['download_p_h3_image'], {
            'default': False, 'folders': [], 'extra_models': list(expected()),
        })
        self.assertEqual({name for name in registry if name.startswith('p-h3-image_')},
                         set(expected()))
        self.assertFalse(any(group['env'] == 'p_h3_image_quant'
                             for group in template.get('swap_groups', [])))

    def test_base_image_is_inherited(self):
        expected_base = 'hearmeman/comfyui-base:cu130-comfy0.44.0-2c75dc1b-torch2.11.0'
        pins = json.loads((REPO / 'pins.json').read_text())
        dockerfile = (REPO / 'Dockerfile').read_text()
        self.assertEqual(pins['base_image'], expected_base)
        self.assertIn(f'ARG BASE_IMAGE={expected_base}\nFROM ${{BASE_IMAGE}}', dockerfile)
        self.assertNotIn('COMFYUI_REF', dockerfile)
        self.assertNotIn('git -C /ComfyUI checkout', dockerfile)
        commands = '\n'.join(line for line in dockerfile.splitlines()
                             if not line.lstrip().startswith('#'))
        self.assertNotIn('/comfyui-approved-ref', commands)

    def test_disabled(self):
        for flag in (None, 'false', ' FALSE ', '0'):
            with self.subTest(flag=flag):
                self.assertFalse(any(n.startswith('p-h3-image_') for n in self.provision(flag)))

    def test_enabled_and_normalization(self):
        for flag in ('true', ' TRUE ', '1', 'yes'):
            with self.subTest(flag=flag):
                self.assert_bundle(self.provision(flag))

    def test_other_flags_independent(self):
        for minimax, viggle in [('true', 'false'), ('false', 'true'), ('true', 'true')]:
            with self.subTest(minimax=minimax, viggle=viggle):
                baseline = self.provision(minimax=minimax, viggle=viggle)
                both = self.provision('true', minimax, viggle)
                self.assert_bundle(both)
                self.assertEqual({k: v for k, v in both.items() if not k.startswith('p-h3-image_')}, baseline)


if __name__ == '__main__':
    unittest.main()
