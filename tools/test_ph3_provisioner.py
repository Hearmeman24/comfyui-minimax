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
    def provision(self, flag=None, minimax='false', viggle='false', with_workflows=False):
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
            copied = {p.relative_to(root / 'workflows').as_posix(): json.loads(p.read_text())
                      for p in (root / 'workflows').rglob('*.json')}
            return (actual, copied) if with_workflows else actual

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
            'default': False, 'folders': ['P-H3 Image'], 'extra_models': list(expected()),
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

    def test_workflow_gate_and_graphs(self):
        from test_provisioner import assert_graph_integrity
        names = {'P-H3 Image/p-h3-image_text_to_image.json',
                 'P-H3 Image/p-h3-image_edit.json'}
        registry = json.loads((REPO / 'src/models_registry.json').read_text())
        for flag, minimax in [(None, 'false'), ('false', 'false'),
                              ('false', 'true'), ('true', 'false'), ('true', 'true')]:
            with self.subTest(flag=flag, minimax=minimax):
                entries, docs = self.provision(flag, minimax, with_workflows=True)
                selected = {k: v for k, v in docs.items() if k.startswith('P-H3 Image/')}
                if flag != 'true':
                    self.assertFalse(selected)
                    self.assertFalse(set(entries) & set(expected()))
                    continue
                self.assertEqual(set(selected), names)
                self.assert_bundle(entries)
                for path, doc in selected.items():
                    self.assertEqual(doc, json.loads((REPO / 'workflows' / path).read_text()))
                    assert_graph_integrity(doc, path, root=True)
                    nodes = doc['nodes']
                    self.assertFalse(any('lora' in n.get('type', '').lower() for n in nodes))
                    loaders = {n['widgets_values'][0]: n for n in nodes
                               if n.get('widgets_values') and isinstance(n['widgets_values'][0], str)
                               and n['widgets_values'][0] in expected()}
                    self.assertEqual(set(loaders), set(expected()))
                    for name, node in loaders.items():
                        self.assertEqual(node['properties']['models'], [{
                            'name': name, 'url': registry[name]['url'],
                            'directory': registry[name]['subdir'],
                        }])
                    self.assertNotIn('example.png', json.dumps(doc))
                    self.assertNotIn('_bf16.safetensors', json.dumps(doc))
                    for name, node in loaders.items():
                        self.assertIn(name, node.get('widgets_values_named', {}).values())
                    images = [n for n in nodes if n.get('type') == 'LoadImage']
                    self.assertEqual(len(images), 2 if path.endswith('_edit.json') else 0)
                    if images:
                        self.assertEqual({n['id']: (n['mode'], n['title']) for n in images}, {
                            4: (0, 'Image to edit (required)'),
                            5: (0, 'Reference image (required)'),
                        })
                    for image in images:
                        self.assertEqual(image['widgets_values'][0], '')
                        self.assertEqual(image['widgets_values_named']['image'], '')

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
