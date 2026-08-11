#!/usr/bin/env python3
"""Workflow-driven model provisioner.

Reads enabled flags (download_minimax_h3), walks the matching workflow folders,
resolves model basenames against models_registry.json, emits a tab-separated
download manifest for hf_download_manager.py, and copies the selected workflow
JSONs into the user's ComfyUI workflows directory, retargeted onto the
requested quant profile.

Source of truth is the workflows tree itself. Adding a model reference to a
workflow without a corresponding registry entry surfaces as a user-supplied
warning (not a hard error here — the CI validator in tools/validate_models.py
enforces registry coverage at merge time).
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path

FLAG_DIRS = {
    "download_minimax_h3": ["MiniMax H3"],
}

# The bundled workflows are written against the int8 profile; selecting a
# different one swaps both what gets downloaded and what the copied
# workflows load. fp8 e4m3 is native on Ada/Hopper/
# Blackwell; NVFP4 only accelerates on Blackwell (sm_120) and is emulated
# everywhere else; int8_convrot runs on anything. The VAEs have one quant
# each and are always downloaded.
#
# The text encoder is the uncensored "heretic" Qwen3-VL-32B rather than
# Comfy-Org's stock one. The stock encoder refuses a large share of the
# prompts this template exists to serve, and the refusal shows up as a
# degraded/empty conditioning rather than an error. Same architecture, same
# loader, same quants — it drops straight into CLIPLoader with type "minimax".
QUANT_PROFILES = {
    "fp8": {
        "fl2va": "minimax_h3_fl2va_pruned_fp8_scaled.safetensors",
        "ref2va": "minimax_h3_ref2va_pruned_fp8_scaled.safetensors",
        "text_encoder": "qwen3vl_32b_h3_ultra_uncensored_heretic_int8_convrot.safetensors",
    },
    "int8": {
        "fl2va": "minimax_h3_fl2va_pruned_int8_convrot.safetensors",
        "ref2va": "minimax_h3_ref2va_pruned_int8_convrot.safetensors",
        "text_encoder": "qwen3vl_32b_h3_ultra_uncensored_heretic_int8_convrot.safetensors",
    },
    "nvfp4": {
        "fl2va": "minimax_h3_fl2va_pruned_fp8_scaled.safetensors",
        "ref2va": "minimax_h3_ref2va_pruned_fp8_scaled.safetensors",
        "text_encoder": "qwen3vl_32b_heretic_minimax_h3_nvfp4.safetensors",
    },
}
QUANTIZED = {b for p in QUANT_PROFILES.values() for b in p.values()}

MODEL_PAT = re.compile(r'"([^"]+\.(?:safetensors|bin|onnx|pth|ckpt))"')


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--registry", required=True)
    ap.add_argument("--workflows-src", required=True)
    ap.add_argument("--workflows-dst", required=True)
    ap.add_argument("--models-root", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--flag", action="append", dest="flags", default=[])
    ap.add_argument("--quant", default="int8", choices=sorted(QUANT_PROFILES))
    args = ap.parse_args()

    registry = json.loads(Path(args.registry).read_text())
    src_root = Path(args.workflows_src)
    dst_root = Path(args.workflows_dst)
    models_root = Path(args.models_root)
    manifest = Path(args.manifest)

    if not args.flags:
        print("[provisioner] no flags enabled — empty manifest, no workflows copied")
        manifest.write_text("")
        return 0

    folders = []
    for f in args.flags:
        if f not in FLAG_DIRS:
            print(f"[provisioner] unknown flag: {f}", file=sys.stderr)
            return 2
        folders.extend(FLAG_DIRS[f])

    refs: dict[str, set[str]] = {}
    workflow_files: list[Path] = []
    for folder in folders:
        d = src_root / folder
        if not d.is_dir():
            print(f"[provisioner] warning: workflow folder missing: {d}")
            continue
        for wf in d.rglob("*.json"):
            workflow_files.append(wf)
            for m in MODEL_PAT.findall(wf.read_text()):
                refs.setdefault(os.path.basename(m), set()).add(str(wf))

    # The workflows name one quant per role — the selected profile decides
    # what is actually pulled, so queue it explicitly instead of relying on
    # every variant being mentioned somewhere in the workflow tree.
    selected = QUANT_PROFILES[args.quant]
    queued: dict[str, dict] = {b: registry[b] for b in selected.values()}
    user_supplied: list[str] = []
    for b in refs:
        if b in QUANTIZED:
            continue
        if b in registry:
            queued[b] = registry[b]
        else:
            user_supplied.append(b)

    # Skip files that already exist on disk above a sanity threshold.
    # Matches the pre-refactor bash behavior (10 MB cutoff catches
    # complete files while still re-fetching obviously-truncated ones).
    MIN_OK_SIZE = 10 * 1024 * 1024
    skipped_existing = []
    lines = []
    for b in sorted(queued):
        entry = queued[b]
        dest = models_root / entry["subdir"] / b
        if dest.is_file() and dest.stat().st_size >= MIN_OK_SIZE:
            skipped_existing.append(b)
            continue
        lines.append(f"{entry['url']}\t{dest}")
    manifest.write_text("\n".join(lines) + ("\n" if lines else ""))

    # Point the copied workflows' loader widgets at the quant that was
    # actually downloaded. Only exact filename widget values are swapped, so
    # the Model Links notes keep listing every variant (which is also what
    # keeps the alternates visible to the ref walk above).
    selected = QUANT_PROFILES[args.quant]
    swaps = {b: selected[role] for p in QUANT_PROFILES.values()
             for role, b in p.items() if b != selected[role]}

    dst_root.mkdir(parents=True, exist_ok=True)
    copied = 0
    for wf in workflow_files:
        rel = wf.relative_to(src_root)
        out = dst_root / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        doc = json.loads(wf.read_text())
        for group in [doc.get("nodes", [])] + [
            sg.get("nodes", [])
            for sg in doc.get("definitions", {}).get("subgraphs", [])
        ]:
            for n in group:
                wv = n.get("widgets_values")
                if isinstance(wv, list):
                    n["widgets_values"] = [
                        swaps.get(v, v) if isinstance(v, str) else v for v in wv
                    ]
                # The frontend's "Missing Models" dialog reads
                # properties.models, not the loader widget. Leave it on the
                # quant we didn't download and the customer is told a file is
                # missing and handed a button that saves it to their own PC.
                for m in (n.get("properties") or {}).get("models") or []:
                    new = swaps.get(m.get("name"))
                    if new:
                        m["name"] = new
                        m["url"] = registry[new]["url"]
        out.write_text(json.dumps(doc, indent=2, ensure_ascii=False))
        copied += 1

    print(f"[provisioner] flags: {','.join(args.flags)}  quant: {args.quant}")
    print(f"[provisioner] workflows copied: {copied}")
    print(f"[provisioner] models queued: {len(lines)} "
          f"({len(skipped_existing)} already on disk, skipped)")
    if user_supplied:
        print(f"[provisioner] {len(user_supplied)} user-supplied LoRA(s) referenced "
              "but not in registry — place them manually or via "
              "CHECKPOINT_IDS_TO_DOWNLOAD/LORAS_IDS_TO_DOWNLOAD:")
        for b in sorted(user_supplied):
            print(f"  - {b}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
