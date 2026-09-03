# MiniMax-H3 video with its own soundtrack, ComfyUI on RunPod

Created by HearmemanAI. Something not working, or a question about a workflow? Ask in
help-and-support on [my Discord](https://discord.gg/ZVWVhT43GW). That is the only place I do
support, and it is also where new releases are announced.

MiniMax-H3 generates the picture and a 32 kHz stereo soundtrack in one pass. Dialogue, room tone,
footsteps and score come out of the same generation, so there is no separate TTS or lip-sync step.
Clips are 768p, 4 to 15 seconds, at 24 fps.

## Before you deploy

Set all of this on the template before you click Deploy, not after.

Click Edit Template and open the environment variables tab. Set `download_minimax_h3` to true. It
is off by default, and a pod without it boots a working but empty ComfyUI, so the workflows open
with blank loader dropdowns and look broken. The full list is in the next section.

Give the pod a network volume of at least 90 GB for the default quantized models. 100 GB is
comfortable and leaves room for your outputs. Full bf16 is much larger: use at least 180 GB, or
200 GB to leave comfortable output space.

If you want your own CivitAI LoRAs or checkpoints on the pod, set `civitai_token` and the ID
variables below. The steps are
[written up on my Discord](https://discord.com/channels/1359855405613715495/1536707221788950708),
and in
[this article](https://civitai.red/articles/12333/how-to-use-hearmemans-civitai-downloader-when-deploying-a-runpod-template).

Then deploy. The first boot takes 10 to 30 minutes. ComfyUI comes up while the models are still
downloading, so you can look around before it finishes. Later deploys on the same network volume
are much faster.

FYI: this template is built for CUDA 13.0 and above.

## Environment variables

| Variable | Default | What it does |
|---|---|---|
| `download_minimax_h3` | false | Downloads the models and installs the six workflows. Set it to true. |
| `minimax_quant` | int8 | Which build to download: int8, fp8, nvfp4, or false for full bf16. You can leave this alone. |
| `LLM_KEY` | empty | Your OpenRouter key. Only the three Auto Prompt workflows use it. |
| `civitai_token` | empty | Your CivitAI API token |
| `CIVITAI_LORAS` | empty | Comma-separated CivitAI version IDs. They go to `models/loras`. |
| `CIVITAI_CHECKPOINTS` | empty | Comma-separated CivitAI version IDs. They go to `models/checkpoints`. |
| `HF_TOKEN` | empty | Optional. Raises your Hugging Face rate limit, which makes a first boot less likely to stall. |

About `minimax_quant`: the default int8 runs natively on every GPU RunPod rents and the workflows
are already set up for it. fp8 is native on 4090, L40, H100, H200 and RTX 50xx cards and emulated
on anything older, which makes it slower there. nvfp4 downloads the same files as fp8 and only
accelerates on RTX 50xx. Set `minimax_quant` to false to download the full
`minimax_h3_fl2va_bf16.safetensors` and `minimax_h3_ref2va_bf16.safetensors` diffusion models
instead. The text encoder, VAEs, and bundled Turbo LoRAs stay the same. Only the profile you ask
for is downloaded and the workflows are pointed at those files for you, so you never touch a
dropdown.

The template also installs the
[MiniMax H3 Latent Upscaler nodes](https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler).
When `download_minimax_h3` is true, its 3D fp16 weight is downloaded to
`models/latent_upscale_models`. Add “Minimax H3 Latent Upscaler (3D)” to a graph to use it; none of
the six bundled workflows enables it automatically.

## Once it is up

Click Connect, then open port 8188 for ComfyUI or port 8888 for JupyterLab. The boot log is at
`/workspace/comfyui.log`.

Open the Workflows tab in ComfyUI. Every workflow carries notes in the graph telling you what it
does and which settings matter, which is a better place to read than this page. The pod also writes
three notes into the top of that same list on first boot: Welcome, Adding Models, and
Troubleshooting. The Welcome note covers writing a prompt H3 responds to, which is most of the
difference between MiniMax's demo reels and a clip that comes out near-silent.

[My other templates](https://docs.google.com/spreadsheets/d/1NfbfZLzE9GIAD5B_y6xjK1IdW95c14oS1JuIG9QihL8/edit)
