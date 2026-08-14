#!/usr/bin/env python3
"""Self-check: a provisioned pod must be internally consistent, per quant.

Drives the shared runtime's provisioner (comfyui-runtime/src/provisioner.py,
pinned by pins.json) against this repo's REAL template.json,
models_registry.json and workflows/, once per `minimax_quant` value: unset,
int8, fp8, FP8, nvfp4, and a garbage value. For each, the workflows copied
to the user's ComfyUI must declare exactly the model files the download
manifest pulled: in the loader widgets AND in each node's
`properties.models`, which is what the ComfyUI frontend's "Missing Models"
dialog reads. A mismatch there tells the customer a file is missing and
offers to download it to their own PC.

This is also the ONLY gate on template.json's `extra_models` key (the turbo
LoRAs no workflow references): the runtime validator ignores the key and the
provisioner only prints an error line at boot, so a typo there is invisible
everywhere else.

Run: python3 tools/test_provisioner.py
Stdlib only, no pytest. Needs template.json + pins.json in the repo root.
"""
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from validate_models import runtime_dir  # noqa: E402

TEXT_ENCODER = "qwen3vl_32b_minimax_h3_int8_convrot.safetensors"

# The two v1.0 builds and the Ref2VA build are referenced by ZERO workflows
# and download only via template.json's extra_models; the FL2VA v0.1 arrives
# via the workflow scan.
BUNDLED_LORAS = [
    "minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors",
    "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors",
    "minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors",
]
TURBO_LORAS = BUNDLED_LORAS + [
    "minimax_h3_fl2v_lightx2v_turbo_4step_v0.1_comfy.safetensors",
]

# (env value or None for unset) -> expected outcome. Replicates the
# provisioner's resolve rule: exact profile name, else lowercased, else warn
# and fall back to the group's default (the deliberate behaviour change vs
# the old provisioner, which exited 2 and left the pod modelless).
CASES = [None, "int8", "fp8", "FP8", "nvfp4", "not-a-quant"]


def load_json(path: Path, hint: str) -> dict:
    try:
        return json.loads(path.read_text())
    except OSError as e:
        raise SystemExit(f"FATAL: cannot read {path.name} ({hint}): {e}")
    except ValueError as e:
        raise SystemExit(f"FATAL: {path.name} is not valid JSON: {e}")


def expected_profile(group: dict, raw) -> str:
    if raw is None:
        return group["default"]
    if raw in group["profiles"]:
        return raw
    low = raw.strip().lower()
    if low in group["profiles"]:
        return low
    return group["default"]


def declared(workflow_dir: Path, quantized: set, registry: dict) -> set:
    """Every managed (quant-swappable) basename the copied workflows claim
    to load, from widgets and properties.models, top level and subgraphs."""
    names = set()
    for wf in workflow_dir.rglob("*.json"):
        doc = json.loads(wf.read_text())
        groups = [doc.get("nodes", [])] + [
            sg.get("nodes", [])
            for sg in doc.get("definitions", {}).get("subgraphs", [])
        ]
        for group in groups:
            for n in group:
                for v in n.get("widgets_values") or []:
                    if isinstance(v, str) and v in quantized:
                        names.add(v)
                for m in (n.get("properties") or {}).get("models") or []:
                    if m.get("name") in quantized:
                        names.add(m["name"])
                        assert m.get("url") == registry[m["name"]]["url"], (
                            f"{wf.name}: {m['name']} declares url "
                            f"{m.get('url')}, registry says "
                            f"{registry[m['name']]['url']}"
                        )
    return names


def main() -> int:
    template = load_json(REPO / "template.json",
                         "written by the migration's slice A; this test only "
                         "goes green once the slices are integrated")
    registry = load_json(REPO / "src" / "models_registry.json", "registry")

    groups = template.get("swap_groups") or []
    assert len(groups) == 1, f"expected exactly one swap group, got {len(groups)}"
    group = groups[0]
    assert group["env"] == "minimax_quant", group["env"]
    assert group["default"] == "int8", group["default"]
    profiles = group["profiles"]
    quantized = {f for p in profiles.values() for f in p.values()}

    # The DiT quant varies by card. The text encoder does not: every profile
    # ships Comfy-Org's stock int8 build, so no quant can pull a second
    # encoder onto the volume.
    for quant, profile in sorted(profiles.items()):
        assert profile["text_encoder"] == TEXT_ENCODER, (
            f"{quant}: text_encoder is {profile['text_encoder']}, "
            f"expected {TEXT_ENCODER}"
        )
    strays = [b for b in registry
              if b.startswith("qwen3vl") and b != TEXT_ENCODER]
    assert not strays, f"registry carries unused text encoders: {sorted(strays)}"
    print(f"✅ every quant profile loads {TEXT_ENCODER}")

    for b in TURBO_LORAS:
        assert b in registry, f"turbo LoRA missing from registry: {b}"
        assert registry[b]["subdir"] == "loras", (
            f"{b}: subdir is {registry[b]['subdir']}, expected loras"
        )
    flag = template["flags"]["download_minimax_h3"]
    assert sorted(flag.get("extra_models", [])) == sorted(BUNDLED_LORAS), (
        f"extra_models must list exactly the {len(BUNDLED_LORAS)} turbo LoRAs "
        f"no workflow references, got {flag.get('extra_models')}"
    )
    print(f"✅ {len(TURBO_LORAS)} turbo LoRAs registered, "
          f"{len(BUNDLED_LORAS)} bundled via extra_models")

    provisioner = runtime_dir() / "src" / "provisioner.py"
    assert provisioner.is_file(), f"no provisioner at {provisioner}"

    manifests: dict = {}
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        for raw in CASES:
            key = expected_profile(group, raw)
            label = "unset" if raw is None else raw
            slug = re.sub(r"[^A-Za-z0-9]", "_", label)
            dst = tmp / f"wf-{slug}"
            manifest = tmp / f"manifest-{slug}.tsv"
            env = dict(os.environ)
            env["download_minimax_h3"] = "true"
            env.pop("minimax_quant", None)
            if raw is not None:
                env["minimax_quant"] = raw
            proc = subprocess.run(
                [sys.executable, str(provisioner),
                 "--template", str(REPO / "template.json"),
                 "--registry", str(REPO / "src" / "models_registry.json"),
                 "--workflows-src", str(REPO / "workflows"),
                 "--workflows-dst", str(dst),
                 "--models-root", str(tmp / f"models-{slug}"),
                 "--manifest", str(manifest)],
                env=env, capture_output=True, text=True,
            )
            assert proc.returncode == 0, (
                f"{label}: provisioner exited {proc.returncode}\n"
                f"{proc.stdout}\n{proc.stderr}"
            )
            lines = [l for l in manifest.read_text().splitlines() if l]
            downloaded = {l.split("\t")[1].rsplit("/", 1)[1] for l in lines}
            manifests[label] = {l.split("\t", 1)[0] for l in lines}

            wanted = set(profiles[key].values())
            got = declared(dst, quantized, registry)
            assert got == wanted, (
                f"{label}: workflows declare {sorted(got)}, "
                f"selected profile {key!r} is {sorted(wanted)}"
            )
            assert wanted <= downloaded, (
                f"{label}: profile files missing from manifest: "
                f"{sorted(wanted - downloaded)}"
            )
            assert set(TURBO_LORAS) <= downloaded, (
                f"{label}: turbo LoRAs missing from manifest: "
                f"{sorted(set(TURBO_LORAS) - downloaded)}"
            )
            if raw is not None and key != raw and key != raw.strip().lower():
                assert "warning: unknown" in proc.stdout, (
                    f"{label}: expected an unknown-quant warning, got:\n"
                    f"{proc.stdout}"
                )
            print(f"✅ {label} -> {key}: workflows and manifest agree on "
                  f"{len(wanted)} files, all {len(TURBO_LORAS)} turbo LoRAs "
                  f"queued")

    # The garbage value must produce the int8 manifest, byte-identical in
    # URL terms, not merely "some default-ish" set.
    assert manifests["not-a-quant"] == manifests["int8"] == manifests["unset"], (
        "unset / int8 / garbage must queue the same URLs"
    )
    print("✅ all quant profiles consistent; garbage falls back to int8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
