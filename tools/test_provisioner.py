#!/usr/bin/env python3
"""Self-check: a provisioned pod must be internally consistent.

For every quant profile, the workflows copied to the user's ComfyUI must
declare exactly the model files the download manifest pulled — in the loader
widgets AND in each node's `properties.models`, which is what the ComfyUI
frontend's "Missing Models" dialog reads. A mismatch there tells the customer
a file is missing and offers to download it to their own PC.

Run: python3 tools/test_provisioner.py
"""
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from workflow_provisioner import (  # noqa: E402
    BUNDLED_LORAS,
    QUANT_PROFILES,
    QUANTIZED,
)

REGISTRY = json.loads((REPO / "src" / "models_registry.json").read_text())

TEXT_ENCODER = "qwen3vl_32b_minimax_h3_int8_convrot.safetensors"

TURBO_LORAS = [
    "minimax_h3_fl2v_lightx2v_turbo_4step_v0.1_comfy.safetensors",
    "minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors",
    "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors",
]


def check_text_encoder() -> None:
    """The DiT quant varies by card. The text encoder does not.

    Every profile loads Comfy-Org's stock int8 build, so no quant can pull a
    second encoder onto the volume or fall off the stock repo.
    """
    for quant, profile in sorted(QUANT_PROFILES.items()):
        assert profile["text_encoder"] == TEXT_ENCODER, (
            f"{quant}: text_encoder is {profile['text_encoder']}, "
            f"expected {TEXT_ENCODER}"
        )
    strays = [
        b for b in REGISTRY
        if b.startswith("qwen3vl") and b != TEXT_ENCODER
    ]
    assert not strays, f"registry carries unused text encoders: {sorted(strays)}"
    print(f"✅ every quant profile loads {TEXT_ENCODER}")


def check_turbo_loras() -> None:
    """All three turbo LoRAs ship, whatever the bundled workflows load.

    The workflows load one of them; the other two are there so the Turbo LoRA
    dropdown can be switched without downloading anything by hand. Nothing in
    the workflow tree references those two, so the provisioner has to bundle
    them explicitly.
    """
    for b in TURBO_LORAS:
        assert b in REGISTRY, f"turbo LoRA missing from registry: {b}"
        assert REGISTRY[b]["subdir"] == "loras", (
            f"{b}: subdir is {REGISTRY[b]['subdir']}, expected loras"
        )
    unreferenced = [b for b in TURBO_LORAS if b not in workflow_refs()]
    assert set(unreferenced) <= set(BUNDLED_LORAS), (
        "turbo LoRAs no workflow references and the provisioner does not "
        f"bundle: {sorted(set(unreferenced) - set(BUNDLED_LORAS))}"
    )
    print(f"✅ {len(TURBO_LORAS)} turbo LoRAs registered, "
          f"{len(BUNDLED_LORAS)} bundled without a workflow reference")


def workflow_refs() -> set[str]:
    """Every model basename the workflow tree names, in any node or note."""
    pat = re.compile(r'"([^"]+\.(?:safetensors|bin|onnx|pth|ckpt))"')
    names = set()
    for wf in (REPO / "workflows").rglob("*.json"):
        for m in pat.findall(wf.read_text()):
            names.add(m.rsplit("/", 1)[-1])
    return names


def declared(workflow_dir: Path) -> set[str]:
    """Every quantized model basename the copied workflows claim to load."""
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
                    if isinstance(v, str) and v in QUANTIZED:
                        names.add(v)
                for m in (n.get("properties") or {}).get("models") or []:
                    if m.get("name") in QUANTIZED:
                        names.add(m["name"])
                        assert m.get("url") == REGISTRY[m["name"]]["url"], (
                            f"{wf.name}: {m['name']} declares url {m.get('url')}, "
                            f"registry says {REGISTRY[m['name']]['url']}"
                        )
    return names


def main() -> int:
    check_text_encoder()
    check_turbo_loras()
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for quant, profile in sorted(QUANT_PROFILES.items()):
            dst = tmp / f"wf-{quant}"
            manifest = tmp / f"manifest-{quant}.tsv"
            subprocess.run(
                [sys.executable, str(REPO / "src" / "workflow_provisioner.py"),
                 "--registry", str(REPO / "src" / "models_registry.json"),
                 "--workflows-src", str(REPO / "workflows"),
                 "--workflows-dst", str(dst),
                 "--models-root", str(tmp / "models"),
                 "--manifest", str(manifest),
                 "--flag", "download_minimax_h3",
                 "--quant", quant],
                check=True, stdout=subprocess.DEVNULL,
            )
            downloaded = {
                line.split("\t")[1].rsplit("/", 1)[1]
                for line in manifest.read_text().splitlines() if line
            }
            wanted = set(profile.values())
            assert declared(dst) == wanted, (
                f"{quant}: workflows declare {sorted(declared(dst))}, "
                f"profile is {sorted(wanted)}"
            )
            assert wanted <= downloaded, (
                f"{quant}: profile files missing from manifest: "
                f"{sorted(wanted - downloaded)}"
            )
            assert set(TURBO_LORAS) <= downloaded, (
                f"{quant}: turbo LoRAs missing from manifest: "
                f"{sorted(set(TURBO_LORAS) - downloaded)}"
            )
            print(f"✅ {quant}: workflows and manifest agree on {len(wanted)} files, "
                  f"all {len(TURBO_LORAS)} turbo LoRAs queued")
    print("✅ all quant profiles consistent")
    return 0


if __name__ == "__main__":
    sys.exit(main())
