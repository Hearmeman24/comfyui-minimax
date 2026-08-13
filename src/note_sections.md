## What is in this template

This template runs MiniMax-H3, which generates the picture and its
soundtrack in one pass. Dialogue, footsteps, music and room tone come
out of the same generation, with nothing to sync afterwards. Clips are
768p, 4 to 15 seconds, at 24 fps.

It comes with five workflows:

- MiniMax - T2V - Custom Prompt: text to video and audio.
- MiniMax - T2V - Auto Prompt: same, but writes the full H3 prompt for
  you from a one-line idea.
- MiniMax - I2V - Custom Prompt: one image to video and audio.
- MiniMax - I2V - Auto Prompt: same, with the prompt written for you.
- video_minimax_h3_r2v: reference-driven, up to 9 images, 3 video clips
  and 3 audio clips.

The two Auto Prompt workflows need an OpenRouter key. Set the LLM_KEY
variable to your key, or paste it into the OpenRouter API Key node in
the graph. Prompt writing is billed by OpenRouter; everything else runs
on your pod.

## Settings you can change

Set these in the environment variables tab. Click Edit Template before
you deploy, or edit the variables on this pod and restart it.

| Variable | Default | What it does |
|---|---|---|
| download_minimax_h3 | unset | Set to true to download the models and install the workflows. Leave it unset and the pod boots with no models. |
| minimax_quant | int8 | Which build of the model to download: int8, fp8 or nvfp4. See below. |
| LLM_KEY | empty | Your OpenRouter key, for the Auto Prompt workflows. |

## Picking a quant

You can skip this. The default int8 runs natively on every GPU RunPod
rents and the workflows are already set up for it.

| minimax_quant | What you get |
|---|---|
| int8 (default) | The int8 model. Works well everywhere. |
| fp8 | The fp8 model. Native on 4090, L40, H100, H200 and RTX 50xx cards. Slower than int8 on older cards. |
| nvfp4 | Same files as fp8. NVFP4 only accelerates on Blackwell cards (RTX 50xx); other GPUs fall back. |

The text encoder is the same int8 build whichever quant you pick. Only
the quant you ask for is downloaded, and the workflows are pointed at
those files automatically, so you never touch a dropdown. If you set a
value that is not on the list, the pod tells you and uses int8.

## Turbo LoRAs

The T2V and I2V workflows sample through a distilled turbo LoRA from
lightx2v. All three of their builds are downloaded, so trying another
one is a dropdown change in the Turbo LoRA node and nothing else.

| LoRA in the dropdown | Steps | Strength | Sigma shift |
|---|---|---|---|
| minimax_h3_fl2v_lightx2v_turbo_4step_v0.1_comfy (the one loaded) | 4 | 0.5 | 12 video / 3 audio |
| minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16 | 4 | 1.0 | 6 video / 3 audio |
| minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16 | 8 | 1.0 | 12 video / 3 audio |

The v0.1 build has its strength baked into the weights, which is why
the node sits at 0.5. The v1.0 builds ship their own alpha instead, so
start them at 1.0, and change the Beta scheduler's step count when you
switch.

## Live previews while sampling

The bundled workflows show an animated preview of the video while it is
sampling. That part has not changed.

For graphs you build from scratch, the animated preview now follows
VideoHelperSuite's own default, which is off. Earlier versions of this
template switched it on for you. To turn it on, open Settings, find the
VHS section, and enable animated previews.
