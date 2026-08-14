# MiniMax-H3: video and audio in one pass

Created by HearmemanAI

[Troubleshooting guide](https://docs.google.com/document/d/1822H-x7AevWz2T_jzMu8-9e5UlQ-zrH0FhCFmQ6FtRc/edit?usp=sharing) if you hit errors.

A ComfyUI pod template built around the open-weights release of **MiniMax-H3**, which generates the
picture and a native 32 kHz stereo soundtrack at the same time. Dialogue, room tone, footsteps, score.
There is no separate TTS or lip-sync stage.

> ⚠️ **Select CUDA 13.0+ in Additional Filters when you deploy.** Hosts that only offer a CUDA 12.8
> driver are not supported from v4 onward.

| | |
|---|---|
| Resolution | **768p only.** The open weights top out here. |
| Clip length | 4 to 15 seconds at 24 fps |
| Audio | 32 kHz stereo, generated with the video |
| Aspect ratios | 21:9, 16:9, 4:3, 1:1, 3:4, 9:16 |
| Dialogue languages | Arabic, Chinese, English, French, German, Italian, Japanese, Korean, Portuguese, Russian, Spanish |

**About 2K:** MiniMax's 2K output comes from their hosted H3-Regenerate-2K API. It is not part of the
open weights and not in this template. Any local setup advertising 2K is upscaling 768p.

## Workflows

| Workflow | Does |
|---|---|
| `MiniMax - T2V - Custom Prompt` | Text to video and audio |
| `MiniMax - T2V - Auto Prompt` | Same, but writes the structured H3 prompt for you from a one-line idea |
| `MiniMax - I2V - Custom Prompt` | One image to video and audio |
| `MiniMax - I2V - Auto Prompt` | Same, with the prompt written for you |
| `video_minimax_h3_r2v` | Reference-driven: up to 9 images, 3 video clips, 3 audio clips, 12 files total |

The four T2V and I2V workflows are mine. They run a turbo LoRA and give you a TAEH3
live preview while sampling, and they use KJNodes, rgthree and VideoHelperSuite, all baked into the
image. `r2v` is ComfyUI's stock template on native H3 nodes only.

**Auto Prompt** needs an [OpenRouter](https://openrouter.ai) key. See the `LLM_KEY` variable below,
or paste it into the "OpenRouter API Key" node in the graph. Prompt rewriting is billed by
OpenRouter, not by this template. Everything else here runs locally.

## Turbo LoRAs

The T2V and I2V workflows sample through a distilled turbo LoRA from
[lightx2v](https://github.com/ModelTC/Minimax-H3-Turbo). All four of their ComfyUI builds are
downloaded, so trying a different one is a dropdown in the Turbo LoRA node and nothing else.

| LoRA in the dropdown | Distilled for | Trained at | Distilled NFE | lightx2v's recommended steps |
|---|---|---|---|---|
| `minimax_h3_fl2v_lightx2v_turbo_4step_v0.1_comfy` (the one loaded) | FL2VA / T2VA | 544p, mixed aspect | 4 | 4 |
| `minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16` | FL2VA / T2VA | 768p, 1344x768 | 4 | 4 |
| `minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16` | FL2VA / T2VA | 544p, mixed aspect | 8 | 8 or 4 |
| `minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16` | Ref2VA | 544p, mixed aspect | 4 | 4 |

The workflows load the FL2VA v0.1 build they were tuned around, and they do not sample at its bare
4 NFE — the Beta scheduler is set to 8 and an `ExtendIntermediateSigmas` node adds more on top of
that schedule. The two v1.0 builds are newer and worth an A/B on your own prompts; change the
Beta scheduler's step count when you switch.

The Ref2VA build is the odd one out: it is distilled for the reference path, and nothing in this
template loads it. `video_minimax_h3_r2v` is ComfyUI's stock graph with no LoRA node in it, so
using this file means adding a `LoraLoaderModelOnly` after the ref2va model loader yourself. It is
downloaded so that it is already on the volume when you do.

Two things move with the file. **Strength:** the FL2VA v0.1 build has its scale baked into the
weights, which is why the node sits at 0.5; the other three carry their own alpha instead, so start
them at 1.0. **Sigma shift:** lightx2v distilled v0.1, the 8-step and the Ref2VA build at 12 video
/ 3 audio, and the 768p one at 6 / 3. Their
[model spec table](https://github.com/ModelTC/Minimax-H3-Turbo#model-specs) is the thing to read
before tuning further.

## Spectrum

[Spectrum](https://github.com/xmarre/ComfyUI-Spectrum-MiniMax-H3) is baked in. It skips some of H3's
transformer evaluations by forecasting the features they would have produced, which makes sampling
faster. Add **Spectrum Apply MiniMax H3** (under `sampling/spectrum`) to a graph to use it. Nothing
uses it unless you add it.

The speedup is not free: forecasting changes the denoising trajectory, so the same prompt and seed
will not give you the same clip it gives you without the node. A/B the seed before you commit to it,
and turn it off for finals. The pack's own README is worth reading before you tune anything.

## Environment variables

| Variable | Description |
|---|---|
| `download_minimax_h3` | Set to `"true"`. Downloads the weights and installs the workflows. |
| `minimax_quant` | `int8` (default), `fp8`, or `nvfp4` |
| `LLM_KEY` | OpenRouter API key, used by the Auto Prompt workflow |
| `civitai_token` | CivitAI token for auto-downloading LoRAs and checkpoints |
| `LORAS_IDS_TO_DOWNLOAD` | Comma-separated CivitAI LoRA version IDs |
| `CHECKPOINT_IDS_TO_DOWNLOAD` | Comma-separated CivitAI checkpoint version IDs |

`download_minimax_h3` is not optional. Leave it unset and the pod boots a healthy ComfyUI with no
models in it, so the workflows open with empty loader dropdowns and look broken.

## Quantization

**Skip this.** The default `int8` runs natively on every GPU RunPod rents and the workflows are
already set up for it.

| `minimax_quant` | Encoder | Transformer | Use on |
|---|---|---|---|
| `int8` (default) | int8, 27 GB | int8, 21 GB | Everything |
| `fp8` | int8, 27 GB | fp8, 21 GB | Ada, Hopper and Blackwell (4090, L40, H100, H200, 5090) |
| `nvfp4` | int8, 27 GB | fp8, 21 GB | Same files as `fp8` |

The encoder is the same int8 build in every case; the quant only picks the transformer. `fp8` is
emulated on Ampere, which makes it slower than the default — on Ampere, leave it alone.

Changing quant is an env var plus a pod restart. Only the quant you ask for is downloaded, and the
workflows are repointed at those files automatically, so you never touch a dropdown.

Budget 82 GB of network volume whichever quant you pick — the four turbo LoRAs are about 8 GB of
that. 100 GB is comfortable with room for outputs.

## Image variants

| Tag | Build |
|---|---|
| `:vN` | CUDA 13.0, torch 2.11.0 cu130. Prebuilt SageAttention wheel, no build on first boot. Needed for native NVFP4. |
| `:vN-cuda12` | Discontinued from v4. Tags v3 and earlier still have it for hosts pinned to a 12.8 driver. |

Both ship **ComfyUI v0.32.0**, pinned rather than tracked, so a tag rebuilds to the same thing later.

## Deploying

1. Set `download_minimax_h3` to `"true"`
2. Deploy. First boot pulls the weights, so expect 10 to 30 minutes.
3. **Connect → port 8188** for ComfyUI, **port 8888** for JupyterLab

Later deployments on the same network volume are much faster.

## Prompting

ComfyUI hands your prompt to H3 verbatim. Nothing rewrites it on the way in, which is why the local
model can feel worse than MiniMax's demo reels. The demos are fed a structured prompt and most people
type a sentence.

H3 wants named sections. For text-to-video, three:

```
integrated_multimodal_description: ...
overall_soundscape: ...
non_diegetic_music: ...
```

A thin `overall_soundscape` gives you a near-silent clip. The audio is generated, not attached, so
describe the ambience, the sounds of the action, the breathing. `non_diegetic_music` wants
instrumentation, tempo and dynamics rather than mood words: "sparse fingerpicked guitar, slow,
dropping to near silence" works, "melancholy and hopeful" does not.

Aim for 350 to 500 words of description. MiniMax's own prompt-writing guides cover shot-cut
timestamps and `<d>` dialogue blocks.

## Troubleshooting

If ComfyUI reports `IMPORT FAILED`, open Manager, click Install missing custom nodes, then Try fix.
Every pack the bundled workflows need is already in the image, so a red node in one of them means
something is wrong with the image rather than your setup.

The image bakes in KJNodes, rgthree, VideoHelperSuite, the OpenRouter pack the Auto Prompt workflows
need, Spectrum, Essentials, Easy-Use, Frame Interpolation, UltimateSDUpscale, Impact Pack, RMBG,
segment-anything-2 and the Manager. Frame Interpolation and segment-anything-2 fetch their own
weights on first use, so expect a wait there.

If the boot log lists a model as "user-supplied", it is a LoRA a workflow references but the
template does not bundle. Drop it into `/workspace/ComfyUI/models/loras/` or pull it with
`LORAS_IDS_TO_DOWNLOAD`.

> **My other templates:**
> [spreadsheet](https://docs.google.com/spreadsheets/d/1NfbfZLzE9GIAD5B_y6xjK1IdW95c14oS1JuIG9QihL8/edit?usp=sharing)
