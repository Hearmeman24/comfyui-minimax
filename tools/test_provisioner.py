#!/usr/bin/env python3
"""Self-check: a provisioned pod must be internally consistent, per profile.

Drives the shared runtime's provisioner (comfyui-runtime/src/provisioner.py,
pinned by pins.json) against this repo's REAL template.json,
models_registry.json and workflows/ through the REAL pre-download hook, across
the supported `minimax_quant` and `minimax_precision` values, mixed precedence,
case normalization, and invalid-value fallback. For each, the workflows copied
to the user's ComfyUI must declare exactly the model files the download manifest
pulled: in the loader widgets AND in each node's
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
BF16_MODELS = {
    "fl2va": "minimax_h3_fl2va_bf16.safetensors",
    "ref2va": "minimax_h3_ref2va_bf16.safetensors",
}
HOOK = REPO / "src" / "hooks" / "pre_download.sh"

# The four v1.0/Ref2VA builds are referenced by ZERO workflows and download
# only via template.json's extra_models; the FL2VA v0.1 arrives via the
# workflow scan.
BUNDLED_LORAS = [
    "minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors",
    "minimax_h3_fl2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors",
    "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors",
    "minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors",
]
TURBO_LORAS = BUNDLED_LORAS + [
    "minimax_h3_fl2v_lightx2v_turbo_4step_v0.1_comfy.safetensors",
]

# label, minimax_quant, minimax_precision, expected internal profile,
# expected warning fragment. Precision=bf16 overrides quant; unset/convrot
# preserves the existing quant selector. Invalid public values keep booting on
# the safe current/default profile with a warning.
CASES = [
    ("unset", None, None, "int8", None),
    ("int8", "int8", None, "int8", None),
    ("fp8", "fp8", None, "fp8", None),
    ("FP8", "FP8", None, "fp8", None),
    ("nvfp4", "nvfp4", None, "nvfp4", None),
    ("invalid-quant", "not-a-quant", None, "int8",
     "warning: unknown minimax_quant"),
    ("convrot", None, "convrot", "int8", None),
    ("convrot-fp8", "fp8", " ConvRot ", "fp8", None),
    ("bf16", None, "bf16", "bf16", None),
    ("bf16-overrides-fp8", "fp8", " BF16 ", "bf16", None),
    ("invalid-precision", "nvfp4", "not-a-precision", "nvfp4",
     "unknown minimax_precision"),
]

HOOK_WRAPPER = r"""
report_warn() { :; }
source "$1"
shift
exec "$@"
"""


def load_json(path: Path, hint: str) -> dict:
    try:
        return json.loads(path.read_text())
    except OSError as e:
        raise SystemExit(f"FATAL: cannot read {path.name} ({hint}): {e}")
    except ValueError as e:
        raise SystemExit(f"FATAL: {path.name} is not valid JSON: {e}")


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
    assert set(profiles) == {"int8", "fp8", "nvfp4", "bf16"}, profiles
    quantized = {f for p in profiles.values() for f in p.values()}
    diffusion_models = {
        profile[role]
        for profile in profiles.values()
        for role in ("fl2va", "ref2va")
    }

    assert {role: profiles["bf16"][role] for role in BF16_MODELS} == (
        BF16_MODELS
    ), profiles["bf16"]
    for role, basename in BF16_MODELS.items():
        assert registry[basename]["subdir"] == "diffusion_models", (
            f"{role}: {basename} must install under diffusion_models"
        )
        assert registry[basename]["url"] == (
            "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/"
            f"diffusion_models/{basename}"
        ), f"{role}: unexpected bf16 URL {registry[basename]['url']}"
    print("✅ bf16 profile points at the two requested Comfy-Org models")

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
    print(f"✅ every model profile loads {TEXT_ENCODER}")

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
    assert HOOK.is_file(), f"no pre-download hook at {HOOK}"

    manifests: dict = {}
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        comfyui_dir = tmp / "comfyui"
        (comfyui_dir / "custom_nodes" / "ComfyUI-Openrouter_node").mkdir(
            parents=True
        )
        persist_root = tmp / "persist"
        persist_root.mkdir()
        for label, raw_quant, raw_precision, key, warning in CASES:
            slug = re.sub(r"[^A-Za-z0-9]", "_", label)
            dst = tmp / f"wf-{slug}"
            manifest = tmp / f"manifest-{slug}.tsv"
            env = dict(os.environ)
            env["download_minimax_h3"] = "true"
            env.pop("minimax_quant", None)
            env.pop("minimax_precision", None)
            env["COMFYUI_DIR"] = str(comfyui_dir)
            env["PERSIST_ROOT"] = str(persist_root)
            if raw_quant is not None:
                env["minimax_quant"] = raw_quant
            if raw_precision is not None:
                env["minimax_precision"] = raw_precision
            proc = subprocess.run(
                ["bash", "-c", HOOK_WRAPPER, "precision-hook", str(HOOK),
                 sys.executable, str(provisioner),
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
            selected_diffusion = downloaded & diffusion_models
            wanted_diffusion = {
                profiles[key]["fl2va"], profiles[key]["ref2va"]
            }
            assert selected_diffusion == wanted_diffusion, (
                f"{label}: queued diffusion files {sorted(selected_diffusion)}, "
                f"expected only {sorted(wanted_diffusion)}"
            )
            assert set(TURBO_LORAS) <= downloaded, (
                f"{label}: turbo LoRAs missing from manifest: "
                f"{sorted(set(TURBO_LORAS) - downloaded)}"
            )
            if warning:
                assert warning in proc.stdout, (
                    f"{label}: expected warning {warning!r}, got:\n{proc.stdout}"
                )
            print(f"✅ {label} -> {key}: workflows and manifest agree on "
                  f"{len(wanted)} files, all {len(TURBO_LORAS)} turbo LoRAs "
                  f"queued")

    # Fallback and aliases must be byte-identical in URL terms, not merely
    # "some default-ish" sets.
    assert (manifests["invalid-quant"] == manifests["int8"] ==
            manifests["unset"] == manifests["convrot"]), (
        "unset / convrot / int8 / invalid quant must queue the same URLs"
    )
    assert manifests["fp8"] == manifests["FP8"] == manifests["convrot-fp8"], (
        "convrot must preserve fp8 and quant matching must ignore case"
    )
    assert manifests["bf16"] == manifests["bf16-overrides-fp8"], (
        "bf16 precision must override minimax_quant"
    )
    assert manifests["invalid-precision"] == manifests["nvfp4"], (
        "invalid precision must preserve the requested current quant"
    )
    print("✅ quant + precision profiles consistent; fallbacks are safe")
    return 0


if __name__ == "__main__":
    sys.exit(main())
