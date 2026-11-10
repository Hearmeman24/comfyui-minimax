#!/usr/bin/env python3
"""Exercise the optional MiniMax H3 Image bundle through the real runtime, offline.

download_minimax_image installs ComfyUI's two stock MiniMax H3 Image workflows
(Comfy-Org/workflow_templates image_minimax_h3_image_t2i / _edit at 0.11.105)
and follows minimax_quant through its own swap group: int8, fp8 and nvfp4 all
load the int8 DiT (there is no fp8 or nvfp4 build), bf16 and false load the
full bf16 DiT. The text encoder is the video template's int8 Qwen3-VL in every
profile, so turning on both flags never puts a second encoder on the volume.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from validate_models import runtime_dir

REPO = Path(__file__).resolve().parents[1]
FOLDER = 'MiniMax H3 Image'
WORKFLOWS = {f'{FOLDER}/MiniMax H3 Image - Text to Image.json',
             f'{FOLDER}/MiniMax H3 Image - Image Edit.json'}
IMAGE = 'https://huggingface.co/Comfy-Org/MiniMax-H3-Image/resolve/main'
INT8 = 'minimax_h3_image_pruned_int8_convrot.safetensors'
BF16 = 'minimax_h3_image_pruned_bf16.safetensors'
TEXT_ENCODER = 'qwen3vl_32b_minimax_h3_int8_convrot.safetensors'
SHARED = {
    TEXT_ENCODER: ('https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/'
                   f'text_encoders/{TEXT_ENCODER}', 'text_encoders'),
    'minimax_h3_image_vae_fp16.safetensors': (
        f'{IMAGE}/vae/minimax_h3_image_vae_fp16.safetensors', 'vae'),
    'qwen3.5_9b_w4a8_gptq.safetensors': (
        'https://huggingface.co/Comfy-Org/Qwen3.5/resolve/main/text_encoders/'
        'qwen3.5_9b_w4a8_gptq.safetensors', 'text_encoders'),
}
DITS = {INT8, BF16}
QUANTS = {None: INT8, 'int8': INT8, 'fp8': INT8, 'nvfp4': INT8, ' FP8 ': INT8,
          'junk': INT8, 'false': BF16, 'bf16': BF16, ' BF16 ': BF16}


def bundle(dit):
    files = dict(SHARED)
    files[dit] = (f'{IMAGE}/diffusion_models/{dit}', 'diffusion_models')
    return {name: (url, f'{sub}/{name}') for name, (url, sub) in files.items()}


def all_nodes(doc):
    yield from doc.get('nodes', [])
    for sg in doc.get('definitions', {}).get('subgraphs', []):
        yield from sg.get('nodes', [])


class MiniMaxImageProvisioning(unittest.TestCase):
    def provision(self, flag=None, quant=None, minimax='false'):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            models = root / 'models'
            manifest = root / 'manifest.tsv'
            env = dict(os.environ)
            for key in ('download_minimax_image', 'minimax_quant',
                        'download_p_h3_image', 'download_viggle_animate'):
                env.pop(key, None)
            env['download_minimax_h3'] = minimax
            if flag is not None:
                env['download_minimax_image'] = flag
            if quant is not None:
                env['minimax_quant'] = quant
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
            queued = {Path(dest).name: (url, str(Path(dest).relative_to(models)))
                      for url, dest in entries}
            self.assertEqual(len(queued), len(entries))
            copied = {p.relative_to(root / 'workflows').as_posix(): json.loads(p.read_text())
                      for p in (root / 'workflows').rglob('*.json')}
            return queued, copied

    def test_configuration(self):
        template = json.loads((REPO / 'template.json').read_text())
        registry = json.loads((REPO / 'src/models_registry.json').read_text())
        self.assertEqual(template['flags']['download_minimax_image'],
                         {'default': False, 'folders': [FOLDER]})
        group = next(g for g in template['swap_groups']
                     if g['flags'] == ['download_minimax_image'])
        self.assertEqual(group['env'], 'minimax_quant')
        self.assertEqual(group['default'], 'int8')
        video = next(g for g in template['swap_groups']
                     if g['flags'] == ['download_minimax_h3'])
        self.assertEqual(set(group['profiles']), set(video['profiles']))
        for quant, profile in group['profiles'].items():
            self.assertEqual(profile, {
                'image_dit': BF16 if quant in ('false', 'bf16') else INT8,
                'text_encoder': TEXT_ENCODER,
            }, quant)
        for name, (url, sub) in bundle(INT8).items() | bundle(BF16).items():
            self.assertEqual(registry[name], {'url': url, 'subdir': sub.split('/')[0]})
        on_disk = {p.relative_to(REPO / 'workflows').as_posix()
                   for p in (REPO / 'workflows' / FOLDER).glob('*.json')}
        self.assertEqual(on_disk, WORKFLOWS)

    def test_source_workflows(self):
        from test_provisioner import assert_graph_integrity
        registry = json.loads((REPO / 'src/models_registry.json').read_text())
        for path in WORKFLOWS:
            with self.subTest(path=path):
                doc = json.loads((REPO / 'workflows' / path).read_text())
                assert_graph_integrity(doc, path, root=True)
                types = {n.get('type') for n in all_nodes(doc)}
                self.assertIn('MiniMaxH3ImageConditioning', types)
                for node in all_nodes(doc):
                    for model in (node.get('properties') or {}).get('models') or []:
                        self.assertIn(model['name'], registry, path)
                        self.assertEqual(model['url'], registry[model['name']]['url'])
                        self.assertEqual(model['directory'], registry[model['name']]['subdir'])
                images = [n for n in all_nodes(doc) if n.get('type') == 'LoadImage']
                self.assertEqual(len(images), 1 if path.endswith('Image Edit.json') else 0)
                for image in images:
                    self.assertEqual(image['widgets_values'][0], '')
                    self.assertEqual(image['widgets_values_named']['image'], '')
                    self.assertEqual(image['title'], 'Image to edit (required)')
                self.assertNotIn(BF16, json.dumps(doc))

    def test_disabled(self):
        for flag in (None, 'false', ' FALSE ', '0'):
            for minimax in ('false', 'true'):
                with self.subTest(flag=flag, minimax=minimax):
                    queued, copied = self.provision(flag, minimax=minimax)
                    self.assertFalse(DITS & set(queued))
                    self.assertNotIn('minimax_h3_image_vae_fp16.safetensors', queued)
                    self.assertFalse(any(p.startswith(f'{FOLDER}/') for p in copied))

    def test_follows_minimax_quant(self):
        for quant, dit in QUANTS.items():
            with self.subTest(quant=quant):
                queued, copied = self.provision('true', quant)
                self.assertEqual(queued, bundle(dit))
                self.assertEqual(set(copied), WORKFLOWS)
                other = (DITS - {dit}).pop()
                for path, doc in copied.items():
                    loaders = [n for n in all_nodes(doc) if n.get('type') == 'UNETLoader']
                    self.assertTrue(loaders, path)
                    for node in loaders:
                        self.assertEqual(node['widgets_values'][0], dit, path)
                        named = node.get('widgets_values_named') or {}
                        if 'unet_name' in named:
                            self.assertEqual(named['unet_name'], dit, path)
                        self.assertEqual(node['properties']['models'], [{
                            'name': dit, 'url': f'{IMAGE}/diffusion_models/{dit}',
                            'directory': 'diffusion_models'}], path)
                    for node in doc['nodes']:
                        if node.get('type') in ('MarkdownNote', 'Note'):
                            continue
                        self.assertNotIn(other, node.get('widgets_values') or [], path)

    def test_with_video_shares_text_encoder(self):
        for quant in ('int8', 'fp8', 'bf16'):
            with self.subTest(quant=quant):
                video, video_wf = self.provision('false', quant, minimax='true')
                both, both_wf = self.provision('true', quant, minimax='true')
                self.assertEqual({k: v for k, v in both.items() if k in video}, video)
                self.assertEqual(set(both) - set(video),
                                 set(bundle(QUANTS[quant])) - {TEXT_ENCODER})
                self.assertEqual(
                    {k: v for k, v in both_wf.items() if not k.startswith(f'{FOLDER}/')},
                    video_wf)
                encoders = {n for n in both if n.startswith('qwen3vl')}
                self.assertEqual(encoders, {TEXT_ENCODER})


if __name__ == '__main__':
    unittest.main()
