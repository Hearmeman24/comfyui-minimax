#!/usr/bin/env python3
"""Self-check: a provisioned pod must be internally consistent, per profile.

Drives the shared runtime's provisioner (comfyui-runtime/src/provisioner.py,
pinned by pins.json) against this repo's REAL template.json,
models_registry.json and workflows/ across every supported `minimax_quant`
value, case/whitespace normalization, and invalid-value fallback. For each, the
workflows copied to the user's ComfyUI must declare exactly the model files the
download manifest pulled: in the loader widgets AND in each node's
`properties.models`, which is what the ComfyUI frontend's "Missing Models"
dialog reads. A mismatch there tells the customer a file is missing and
offers to download it to their own PC. It also checks the source workflow
links and the tuned loader/scheduler defaults before the provisioner rewrites
copies for another quant profile.

This is also the ONLY gate on template.json's `extra_models` key (the turbo
LoRAs and latent upscaler no workflow references): the runtime validator
ignores the key and the provisioner only prints an error line at boot, so a
typo there is invisible everywhere else.

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

# The refreshed workflows use the 8-step 768p builds. These six alternates are
# still bundled for dropdown switching, so they download only through
# template.json's extra_models.
KJ_TURBO = "minimax_h3_fl2v_lightx2v_turbo_4step_v0.1_comfy.safetensors"
BUNDLED_LORAS = [
    KJ_TURBO,
    "minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors",
    "minimax_h3_fl2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors",
    "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors",
    "minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors",
    "minimax_h3_ref2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors",
]
TURBO_LORAS = BUNDLED_LORAS
LATENT_UPSCALER = "minimax_h3_latent_upscaler_3d_fp16.safetensors"
LATENT_UPSCALER_URL = (
    "https://huggingface.co/LBH-123-AI/Minimax_h3_latent_Upscaler/resolve/"
    "main/minimax_h3_latent_upscaler_3d_fp16.safetensors"
)
LATENT_UPSCALER_NODE = (
    "https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler.git|"
    "d7c01b9011f2e8439493f6c02c29995a27df276f"
)
EXTRA_MODELS = BUNDLED_LORAS + [LATENT_UPSCALER]
FL2VA_INT8 = "minimax_h3_fl2va_pruned_int8_convrot.safetensors"
REF2VA_INT8 = "minimax_h3_ref2va_pruned_int8_convrot.safetensors"
FL2VA_TURBO_8STEP = (
    "minimax_h3_fl2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors"
)
REF2VA_TURBO_8STEP = (
    "minimax_h3_ref2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors"
)
SOURCE_WORKFLOW_DEFAULTS = {
    "MiniMax - I2V - Auto Prompt.json":
        (FL2VA_INT8, FL2VA_TURBO_8STEP, 0.8, True),
    "MiniMax - I2V - Custom Prompt.json":
        (FL2VA_INT8, FL2VA_TURBO_8STEP, 0.8, True),
    "MiniMax - R2V - Auto Prompt.json":
        (REF2VA_INT8, REF2VA_TURBO_8STEP, 0.85, False),
    "MiniMax - T2V - Auto Prompt.json":
        (FL2VA_INT8, FL2VA_TURBO_8STEP, 0.8, False),
    "MiniMax - T2V - Custom Prompt.json":
        (FL2VA_INT8, FL2VA_TURBO_8STEP, 0.8, False),
}

# label, minimax_quant, expected profile, expected warning fragment. The false
# profile selects full bf16; invalid values keep booting on int8 with a warning.
CASES = [
    ("unset", None, "int8", None),
    ("int8", "int8", "int8", None),
    ("fp8", "fp8", "fp8", None),
    ("FP8", "FP8", "fp8", None),
    ("nvfp4", "nvfp4", "nvfp4", None),
    ("false", "false", "false", None),
    ("FALSE", "FALSE", "false", None),
    ("false-whitespace", " false ", "false", None),
    ("invalid-quant", "not-a-quant", "int8",
     "warning: unknown minimax_quant"),
]


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


def normalized_links(raw_links: list) -> list:
    """Normalize root array links and subgraph object links."""
    result = []
    for link in raw_links or []:
        if isinstance(link, list):
            assert len(link) >= 5, f"malformed root link: {link}"
            result.append({
                "id": link[0], "origin_id": link[1],
                "origin_slot": link[2], "target_id": link[3],
                "target_slot": link[4],
            })
        else:
            result.append(link)
    return result


def assert_graph_integrity(graph: dict, label: str, root: bool = False) -> None:
    """Every serialized link must resolve through both endpoint slots."""
    nodes = graph.get("nodes") or []
    node_ids = [node["id"] for node in nodes]
    assert len(node_ids) == len(set(node_ids)), f"{label}: duplicate node id"
    by_id = {node["id"]: node for node in nodes}
    pseudo_ids = {
        node["id"] for node in (graph.get("inputNode"),
                                graph.get("outputNode"))
        if isinstance(node, dict)
    }

    links = normalized_links(graph.get("links") or [])
    link_ids = [link["id"] for link in links]
    assert len(link_ids) == len(set(link_ids)), f"{label}: duplicate link id"
    by_link_id = {link["id"]: link for link in links}

    for link in links:
        link_id = link["id"]
        origin_id = link["origin_id"]
        target_id = link["target_id"]
        assert origin_id in by_id or origin_id in pseudo_ids, (
            f"{label}: link {link_id} has missing origin {origin_id}"
        )
        assert target_id in by_id or target_id in pseudo_ids, (
            f"{label}: link {link_id} has missing target {target_id}"
        )
        if origin_id in by_id:
            outputs = by_id[origin_id].get("outputs") or []
            slot = link["origin_slot"]
            assert isinstance(slot, int) and slot < len(outputs), (
                f"{label}: link {link_id} has invalid origin slot {slot}"
            )
            assert link_id in (outputs[slot].get("links") or []), (
                f"{label}: origin {origin_id}:{slot} omits link {link_id}"
            )
        if target_id in by_id:
            inputs = by_id[target_id].get("inputs") or []
            slot = link["target_slot"]
            assert isinstance(slot, int) and slot < len(inputs), (
                f"{label}: link {link_id} has invalid target slot {slot}"
            )
            assert inputs[slot].get("link") == link_id, (
                f"{label}: target {target_id}:{slot} omits link {link_id}"
            )

    for node in nodes:
        for output in node.get("outputs") or []:
            for link_id in output.get("links") or []:
                assert link_id in by_link_id, (
                    f"{label}: node {node['id']} output references missing "
                    f"link {link_id}"
                )
        for input_slot in node.get("inputs") or []:
            link_id = input_slot.get("link")
            assert link_id is None or link_id in by_link_id, (
                f"{label}: node {node['id']} input references missing "
                f"link {link_id}"
            )

    if root and node_ids:
        assert graph["last_node_id"] >= max(node_ids), (
            f"{label}: last_node_id is below a live node id"
        )
    if root and link_ids:
        assert graph["last_link_id"] >= max(link_ids), (
            f"{label}: last_link_id is below a live link id"
        )


def assert_source_workflows(registry: dict) -> None:
    workflow_root = REPO / "workflows" / "MiniMax H3"
    for workflow_file in workflow_root.glob("*.json"):
        doc = load_json(workflow_file, "workflow")
        assert_graph_integrity(doc, workflow_file.name, root=True)
        for subgraph in doc.get("definitions", {}).get("subgraphs", []):
            assert_graph_integrity(
                subgraph,
                f"{workflow_file.name}/{subgraph.get('name', subgraph['id'])}",
            )

    for name, defaults in SOURCE_WORKFLOW_DEFAULTS.items():
        diffusion, lora, strength, has_input_image = defaults
        doc = load_json(workflow_root / name, "workflow defaults")
        loaders = [node for node in doc["nodes"]
                   if node.get("type") == "UNETLoader"]
        assert len(loaders) == 1, f"{name}: expected one UNETLoader"
        loader = loaders[0]
        assert loader.get("widgets_values") == [diffusion, "default"], (
            f"{name}: wrong default diffusion model: "
            f"{loader.get('widgets_values')}"
        )
        assert (loader.get("widgets_values_named") or {}).get(
            "unet_name") == diffusion, f"{name}: named UNET value drifted"
        assert (loader.get("properties") or {}).get("models") == [{
            "name": diffusion,
            "url": registry[diffusion]["url"],
            "directory": registry[diffusion]["subdir"],
        }], f"{name}: UNET properties.models does not match the registry"

        lora_nodes = [node for node in doc["nodes"]
                      if node.get("type") == "LoraLoaderModelOnly"]
        assert len(lora_nodes) == 1, f"{name}: expected one Turbo LoRA loader"
        lora_node = lora_nodes[0]
        assert lora_node.get("widgets_values") == [lora, strength], (
            f"{name}: Turbo LoRA does not match its 8-step schedule"
        )
        assert lora_node.get("widgets_values_named") == {
            "lora_name": lora, "strength_model": strength,
        }, f"{name}: named Turbo LoRA values drifted"

        schedulers = [node for node in doc["nodes"]
                      if node.get("type") == "BasicScheduler"]
        assert len(schedulers) == 1, f"{name}: expected one BasicScheduler"
        assert schedulers[0].get("widgets_values") == ["simple", 8, 1], (
            f"{name}: expected the tuned 8-step simple schedule"
        )
        assert not any(node.get("type") == "BetaSamplingScheduler"
                       for node in doc["nodes"]), (
            f"{name}: obsolete BetaSamplingScheduler remains"
        )

        if has_input_image:
            image_nodes = [node for node in doc["nodes"]
                           if node.get("type") == "LoadImage"]
            assert len(image_nodes) == 1, f"{name}: expected one LoadImage"
            assert image_nodes[0].get("widgets_values") == [
                "your_input_image.png", "image"
            ], f"{name}: stale local image default remains"

    print("✅ all six workflow graphs have internally consistent links")
    print("✅ five refreshed workflows use the int8 defaults and 8-step Turbo")


def main() -> int:
    template = load_json(REPO / "template.json",
                         "written by the migration's slice A; this test only "
                         "goes green once the slices are integrated")
    registry = load_json(REPO / "src" / "models_registry.json", "registry")

    assert_source_workflows(registry)

    groups = template.get("swap_groups") or []
    assert len(groups) == 1, f"expected exactly one swap group, got {len(groups)}"
    group = groups[0]
    assert group["env"] == "minimax_quant", group["env"]
    assert group["default"] == "int8", group["default"]
    profiles = group["profiles"]
    assert set(profiles) == {"int8", "fp8", "nvfp4", "false"}, profiles
    quantized = {f for p in profiles.values() for f in p.values()}
    diffusion_models = {
        profile[role]
        for profile in profiles.values()
        for role in ("fl2va", "ref2va")
    }

    assert {role: profiles["false"][role] for role in BF16_MODELS} == (
        BF16_MODELS
    ), profiles["false"]
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
    for b in BUNDLED_LORAS:
        if b == KJ_TURBO:
            assert registry[b]["url"] == (
                "https://huggingface.co/Kijai/MiniMax-H3_comfy/resolve/main/"
                f"loras/{b}"
            ), f"{b}: unexpected Kijai URL {registry[b]['url']}"
            continue
        assert registry[b]["url"] == (
            "https://huggingface.co/lightx2v/Minimax-h3-Turbo/resolve/main/"
            f"{b}"
        ), f"{b}: unexpected LightX2V URL {registry[b]['url']}"
    flag = template["flags"]["download_minimax_h3"]
    assert LATENT_UPSCALER in registry, "latent upscaler missing from registry"
    assert registry[LATENT_UPSCALER] == {
        "url": LATENT_UPSCALER_URL,
        "subdir": "latent_upscale_models",
        "min_size_mb": 650,
    }, f"unexpected latent upscaler registry entry: {registry[LATENT_UPSCALER]}"
    assert sorted(flag.get("extra_models", [])) == sorted(EXTRA_MODELS), (
        f"extra_models must list exactly the {len(EXTRA_MODELS)} models no "
        f"workflow references, got {flag.get('extra_models')}"
    )
    print(f"✅ {len(TURBO_LORAS)} turbo LoRAs registered, "
          f"{len(BUNDLED_LORAS)} bundled via extra_models")

    custom_nodes = template["custom_nodes"]
    assert custom_nodes["target"] == "image", custom_nodes
    assert custom_nodes.get("repos") == [LATENT_UPSCALER_NODE], (
        f"latent upscaler node must be reproducibly pinned, got {custom_nodes}"
    )
    print("✅ latent upscaler model destination and custom-node pin are exact")

    provisioner = runtime_dir() / "src" / "provisioner.py"
    assert provisioner.is_file(), f"no provisioner at {provisioner}"

    manifests: dict = {}
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        for label, raw_quant, key, warning in CASES:
            slug = re.sub(r"[^A-Za-z0-9]", "_", label)
            dst = tmp / f"wf-{slug}"
            manifest = tmp / f"manifest-{slug}.tsv"
            env = dict(os.environ)
            env["download_minimax_h3"] = "true"
            env.pop("minimax_quant", None)
            if raw_quant is not None:
                env["minimax_quant"] = raw_quant
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
            destinations = {
                Path(line.split("\t")[1]).name:
                    Path(line.split("\t")[1])
                for line in lines
            }
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
            expected_upscaler_path = (
                tmp / f"models-{slug}" / "latent_upscale_models" /
                LATENT_UPSCALER
            )
            assert destinations.get(LATENT_UPSCALER) == expected_upscaler_path, (
                f"{label}: latent upscaler destination is "
                f"{destinations.get(LATENT_UPSCALER)}, expected "
                f"{expected_upscaler_path}"
            )
            if warning:
                assert warning in proc.stdout, (
                    f"{label}: expected warning {warning!r}, got:\n{proc.stdout}"
                )
            print(f"✅ {label} -> {key}: workflows and manifest agree on "
                  f"{len(wanted)} files, all {len(TURBO_LORAS)} turbo LoRAs "
                  f"and the latent upscaler queued")

    # Fallback and aliases must be byte-identical in URL terms, not merely
    # "some default-ish" sets.
    assert (manifests["invalid-quant"] == manifests["int8"] ==
            manifests["unset"]), (
        "unset / int8 / invalid quant must queue the same URLs"
    )
    assert manifests["fp8"] == manifests["FP8"], (
        "quant matching must ignore case"
    )
    assert (manifests["false"] == manifests["FALSE"] ==
            manifests["false-whitespace"]), (
        "false must select bf16 regardless of case or surrounding whitespace"
    )
    print("✅ all minimax_quant profiles consistent; fallbacks are safe")
    return 0


if __name__ == "__main__":
    sys.exit(main())
