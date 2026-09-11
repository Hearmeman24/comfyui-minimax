#!/usr/bin/env python3
"""Exercise opt-in Viggle provisioning through the real pinned runtime (no downloads)."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

REPO = Path(__file__).resolve().parents[1]
VIGGLE_NODE = ('https://github.com/Saganaki22/ComfyUI-Viggle-Animate-H3.git|'
               '77425b6f955e98e96c5e0dc6ca199dc62e63ad8a')
RIFE_URL = ('https://github.com/Fannovel16/ComfyUI-Frame-Interpolation/'
            'releases/download/models/rife426.pth')
VIGGLE = {
    'Viggle-Animate-pruned_rank8_bf16.safetensors': 'diffusion_models',
    'viggle_animate_dmd_lora.safetensors': 'loras',
    'fixed_embed_fwd_anyframe.safetensors': 'text_cond',
    'minimax_h3_video_vae_fp16.safetensors': 'vae',
    'rife426.pth': 'frame_interpolation',
}


def check(provisioner):
    registry = json.loads((REPO / 'src/models_registry.json').read_text())
    template = json.loads((REPO / 'template.json').read_text())
    assert registry['rife426.pth'] == {
        'url': RIFE_URL, 'subdir': 'frame_interpolation', 'min_size_mb': 20,
    }
    assert 'text_cond' in template['extra_model_paths']
    assert VIGGLE_NODE in template['custom_nodes']['repos']
    # A 3.7 MB frozen embedding must not inherit the default 10 MB skip floor.
    assert 0 < registry['fixed_embed_fwd_anyframe.safetensors']['min_size_mb'] < 3.5
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        baseline = None
        for label, flag, minimax, repeat in [
            ('off', None, False, False), ('false', 'false', False, False),
            ('viggle', 'true', False, False), ('normalized', ' TRUE ', False, False),
            ('minimax', 'false', True, False), ('both', 'true', True, False),
            ('existing', 'true', False, True),
            ('partial-rife', 'true', False, True),
            ('duplicate', 'true', False, False),
        ]:
            models = root / label / 'models'
            workflows = root / label / 'workflows'
            manifest = root / label / 'manifest.tsv'
            models.mkdir(parents=True)
            if repeat:
                # Sparse files exercise the provisioner's actual skip-existing floor.
                for name, subdir in VIGGLE.items():
                    path = models / subdir / name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    floor = registry[name].get('min_size_mb', 10) * 1024 * 1024
                    with path.open('wb') as f:
                        f.truncate(floor + 1)
            if label == 'partial-rife':
                (models / 'frame_interpolation/rife426.pth').write_bytes(b'incomplete')
            template_path = REPO / 'template.json'
            if label == 'duplicate':
                # A repeated extra must still emit only one download destination.
                duplicate = json.loads(template_path.read_text())
                duplicate['flags']['download_viggle_animate']['extra_models'].append('rife426.pth')
                template_path = root / label / 'template.json'
                template_path.write_text(json.dumps(duplicate))
            env = dict(os.environ)
            env.pop('download_viggle_animate', None)
            env.pop('minimax_quant', None)
            env['download_minimax_h3'] = str(minimax).lower()
            if flag is not None:
                env['download_viggle_animate'] = flag
            result = subprocess.run([
                sys.executable, str(provisioner), '--template', str(template_path),
                '--registry', str(REPO / 'src/models_registry.json'),
                '--workflows-src', str(REPO / 'workflows'), '--workflows-dst', str(workflows),
                '--models-root', str(models), '--manifest', str(manifest),
            ], env=env, text=True, capture_output=True)
            assert result.returncode == 0, result.stdout + result.stderr
            assert '[provisioner] error:' not in result.stdout, result.stdout
            entries = [line.split('\t')[:2] for line in manifest.read_text().splitlines() if line]
            names = {Path(dest).name for _, dest in entries}
            assert len(entries) == len(names), (label, 'duplicate model downloads', entries)
            copied = {str(p.relative_to(workflows)): p.read_bytes()
                      for p in workflows.rglob('*.json')}
            if label in ('off', 'false', 'existing'):
                assert not names, (label, names)
            elif label == 'partial-rife':
                assert names == {'rife426.pth'}, (label, names)
                assert entries == [[RIFE_URL, str(models / 'frame_interpolation/rife426.pth')]]
            elif not minimax:
                assert names == set(VIGGLE), (label, names)
                for url, dest in entries:
                    name = Path(dest).name
                    assert Path(dest) == models / VIGGLE[name] / name
                    assert url == registry[name]['url']
            elif label == 'minimax':
                baseline = (names, copied)
                assert not (set(VIGGLE) - {'minimax_h3_video_vae_fp16.safetensors'}) & names
            else:
                assert names == baseline[0] | set(VIGGLE), names
                assert copied == baseline[1], 'Viggle must not change copied workflows'
                assert len(entries) == len(names), 'shared VAE must be deduplicated'
            if not minimax:
                assert not copied, 'Viggle must never install workflows'
            print(f'✅ Viggle {label}: models, paths, and workflow isolation agree')


if __name__ == '__main__':
    from validate_models import runtime_dir
    check(runtime_dir() / 'src/provisioner.py')
