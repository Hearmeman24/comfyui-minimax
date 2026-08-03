# Created by HearmemanAI https://www.hearmemanai.com

[![Sponsor](https://readme.cash/i/pift670zt5.svg)](https://readme.cash/c/pift670zt5)

# Troubleshooting guide in case you encounter any errors:
[click here](https://docs.google.com/document/d/1822H-x7AevWz2T_jzMu8-9e5UlQ-zrH0FhCFmQ6FtRc/edit?usp=sharing)
---

# MiniMax-H3: video and audio in one pass

A ComfyUI pod template built around one model, the open-weights release of **MiniMax-H3**. H3
generates the picture and a native 32 kHz stereo soundtrack at the same time: dialogue, room tone,
footsteps, score. There is no separate TTS or lip-sync stage.

> ⚠️ **Deploy with CUDA 13.0+ selected in Additional Filters.** The default image is built on CUDA
> 13 with torch cu130. If your host only offers CUDA 12.8, use the `-cuda12` tag instead (see below).

---

## What you actually get

| | |
|---|---|
| Resolution | **768p only.** The open weights top out here. |
| Clip length | 4 to 15 seconds at 24 fps |
| Audio | 32 kHz stereo, generated with the video |
| Aspect ratios | 21:9, 16:9, 4:3, 1:1, 3:4, 9:16 |
| Dialogue languages | Arabic, Chinese, English, French, German, Italian, Japanese, Korean, Portuguese, Russian, Spanish |

> **About 2K:** MiniMax's 2K output comes from their hosted H3-Regenerate-2K API. It is not part of
> the open weights and it is not in this template. Any local setup advertising 2K is upscaling 768p.

### Bundled workflows

Three, all built on ComfyUI's native H3 nodes:

| Workflow | Does |
|---|---|
| `video_minimax_h3_t2v` | Text to video and audio |
| `video_minimax_h3_i2v` | One image to video and audio |
| `video_minimax_h3_r2v` | Reference-driven: up to 9 images, 3 video clips of 2 to 15s each, 3 audio clips, 12 files total |

---

## ⚙️ Environment Variables

Set the flags you want to **`true`** (lowercase, as strings).

| Variable | Description |
|---|---|
| `download_minimax_h3` | Downloads the H3 weights and copies the three workflows into ComfyUI |
| `minimax_quant` | Which quantization to pull: `int8` (default), `fp8`, or `nvfp4` |

`minimax_quant` also retargets the loader widgets in the copied workflows, so what you open already
points at the files that were downloaded. You never touch a dropdown.

### CivitAI (unchanged)

| Variable | Description |
|---|---|
| `civitai_token` | Your CivitAI token (auto-download LoRAs and Checkpoints) |
| `LORAS_IDS_TO_DOWNLOAD` | Comma-separated CivitAI LoRA version IDs |
| `CHECKPOINT_IDS_TO_DOWNLOAD` | Comma-separated CivitAI Checkpoint version IDs |

👉 [CivitAI Downloader README](https://github.com/Hearmeman24/CivitAI_Downloader/blob/main/README.md)

---

## Picking a quantization

**You can skip this section.** The default is `int8`, it runs natively on every GPU RunPod rents, and
it is what the bundled workflows are set up for. Nothing to configure.

H3 ships as two pieces: a 21 GB diffusion transformer, and a Qwen3-VL-32B text encoder that is
bigger than the transformer. The quant you pick applies to both.

| `minimax_quant` | Text encoder | Diffusion transformer | Use it on |
|---|---|---|---|
| `int8` (default) | int8, 27 GB | int8, 21 GB each | Everything. Ampere, Ada, Hopper, Blackwell. int8 needs no special hardware. |
| `fp8` | int8, 27 GB | fp8 scaled, 21 GB each | Ada (RTX 4090, L40) and Hopper (H100, H200), which run FP8 e4m3 natively. Emulated on Ampere. |
| `nvfp4` | NVFP4 AWQ, 16 GB | fp8 scaled, 21 GB each | Blackwell only (RTX 5090, PRO 6000, B200). The encoder is about 40% smaller than the int8 one. Emulated, and slower, on anything older. |

The other two exist because `int8` is the safe choice, not the fastest one on every card. If you know
you rented Blackwell, `nvfp4` is worth trying. If you rented Ampere, leave it alone: both alternatives
would emulate and you would end up slower than the default.

This template ships the Comfy-Org weights, where NVFP4 is published for the text encoder only. That
is why the `nvfp4` profile pairs an NVFP4 encoder with the FP8 transformer rather than going NVFP4 on
both. Community NVFP4 transformer quants do exist elsewhere; they aren't bundled here.

**Switching quant is an env var change and a pod restart, not a dropdown.** Only the quant you asked
for gets downloaded, so the other variants are not sitting on disk waiting. Change `minimax_quant`,
restart, and the pod fetches the new files and repoints the workflows at them.

The template pulls both transformers, one for text and image to video, one for reference to video.
Budget roughly 75 GB of network volume for `int8` or `fp8`, and 64 GB for `nvfp4`. Add room for
outputs on top of that; 100 GB is comfortable.

The reference deployment for this template ran on an H200 with 80 GB. ComfyUI loads the text encoder
and the transformer in separate passes rather than holding both at once, so smaller cards can work.
But 768p at 15 seconds is not a 24 GB job, and I have not benchmarked where the floor actually sits.

---

## Image variants

| Tag | Build |
|---|---|
| `:vN` | CUDA 13.0 with torch 2.11.0 cu130. The default. Ships a prebuilt SageAttention wheel, so there is no five-minute build on first boot. Required for NVFP4 to run natively. |
| `:vN-cuda12` | CUDA 12.8 with stable torch cu128, for hosts pinned to a 12.8 driver. No NVFP4, and SageAttention compiles from source at boot. |

Both come from the same Dockerfile with four build-args overridden. Both ship **ComfyUI v0.30.1**,
pinned rather than tracked, so a given tag rebuilds to the same thing later. H3 support landed in
v0.30.0, and the image refuses to build below that.

---

## 🚀 Deploying

1. Set `download_minimax_h3` to `"true"`. Leave `minimax_quant` alone unless you've read the section above.
2. Click **Deploy**.
3. Wait for setup. The weights are large, so expect 10 to 30 minutes depending on network.
4. Later deployments off the same network volume are much faster.

Step 1 is not optional. Leave `download_minimax_h3` unset and the pod boots a perfectly healthy
ComfyUI with no models in it, so the workflows open with empty loader dropdowns and look broken. The
boot log says which flag it was waiting for.

## 🌐 Accessing ComfyUI
1. Click **Connect**
2. Open **port 8188**

## 📓 Accessing JupyterLab
1. Click **Connect**
2. Open **port 8888**

---

## Prompting H3

ComfyUI hands your prompt to H3 verbatim. Nothing rewrites it on the way in, which is why the local
model can feel worse than MiniMax's demo reels: the demos are fed a structured prompt, and most
people type a sentence.

H3 wants named sections. For text-to-video that is three of them:

```
integrated_multimodal_description: ...
overall_soundscape: ...
non_diegetic_music: ...
```

Two things that catch people out:

- A thin `overall_soundscape` gives you a near-silent clip. The audio is generated, not attached, so
  describe the ambience, the physical sounds of the action, the breathing.
- `non_diegetic_music` wants instrumentation, tempo and dynamics, not mood words. "Sparse
  fingerpicked guitar, slow, dropping to near silence" works. "Melancholy and hopeful" does not.

Aim for 350 to 500 words in the description. MiniMax's own prompt-writing guides cover the rest of
the format, including shot-cut timestamps and `<d>` dialogue blocks.

---

## 🛠️ Troubleshooting

### `IMPORT FAILED` in ComfyUI
1. Open the **Manager** menu
2. Click **Install missing custom nodes**
3. Click **Try fix**

The H3 nodes are part of ComfyUI core, so the three bundled workflows need no custom node pack at
all. If one of them shows a red node, something is wrong with the image rather than with your setup.

The image does bake in a small general-purpose toolkit for building your own graphs on top of H3:
KJNodes, rgthree, VideoHelperSuite, Essentials, Easy-Use, Frame Interpolation, UltimateSDUpscale,
Impact Pack, RMBG, and segment-anything-2, plus the Manager itself. Anything further you install
through the Manager persists on your network volume.

Two of those fetch their own model weights the first time you use them rather than at build time.
Frame Interpolation pulls its interpolation checkpoint, and segment-anything-2 pulls SAM2. Neither is
part of the bundled model download, so expect a wait on first use and expect it again after a pod
rebuild.

### User-supplied LoRAs
If a workflow references a LoRA that isn't bundled, the boot log lists it as "user-supplied". Drop it
into `/workspace/ComfyUI/models/loras/` manually or via `LORAS_IDS_TO_DOWNLOAD`.

> ℹ️ **For my other templates**:
> [Click HERE](https://docs.google.com/spreadsheets/d/1NfbfZLzE9GIAD5B_y6xjK1IdW95c14oS1JuIG9QihL8/edit?usp=sharing)

[![ko-fi](https://ko-fi.com/img/githubbutton_sm.svg)](https://ko-fi.com/W7W81FM4M1)
