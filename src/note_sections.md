## What is in this template

This template runs MiniMax-H3, which generates the picture and its
soundtrack in one pass. Dialogue, footsteps, music and room tone come
out of the same generation, with nothing to sync afterwards. Clips are
768p, 4 to 15 seconds, at 24 fps.

It comes with six workflows:

- MiniMax - T2V - Custom Prompt: text to video and audio.
- MiniMax - T2V - Auto Prompt: same, but writes the full H3 prompt for
  you from a one-line idea.
- MiniMax - I2V - Custom Prompt: one image to video and audio.
- MiniMax - I2V - Auto Prompt: same, with the prompt written for you.
- MiniMax - R2V - Auto Prompt: reference-driven, with the reference
  manager and the prompt written for you. See below.
- video_minimax_h3_r2v: the stock reference graph, up to 9 images, 3
  video clips and 3 audio clips.

The three Auto Prompt workflows need an OpenRouter key. Set the LLM_KEY
variable to your key, or paste it into the OpenRouter API Key node in
the graph. Prompt writing is billed by OpenRouter; everything else runs
on your pod.

## The R2V reference manager

Reference-to-video is the hardest of the three to prompt by hand. You
are not describing a scene, you are telling H3 which parts of which
reference to keep and which to throw away, and it wants that written
out reference by reference in named sections.

MiniMax - R2V - Auto Prompt does that part for you. The MiniMax
References Manager node in that graph has its own panel: click it, add
your images, video clips and audio clips, and the panel handles the tags
H3 expects (<Picture 1>, <Video 1>, <Audio 1>) and the order they go in.
Then write one line saying what you actually want in the direction box,
and the node writes the full six-section prompt.

Two things worth knowing:

The job_type setting picks which register it writes in. "standard"
writes a scene. "replacement" swaps one thing in a reference video for
the thing in a reference image, which is a different prompt shape
entirely. "auto" is the default and decides for you, and it only bothers
deciding when you have given it at least one video and one image.

Video references are resampled to 24 fps, because that is what H3
generates at. A 30 fps clip keeps its real duration, not its frame
count.

The Generated Prompt panel shows you what the node wrote and sent. If a
clip surprises you, read that before you change anything else. The node
also has a second output called debug, which nothing is wired to in the
shipped graph: connect it to a Display Any node and it reports the
model, the settings and the reference list it used, and on auto it tells
you which register it resolved to.

## Writing a prompt H3 responds to

ComfyUI hands your prompt to H3 exactly as you typed it. Nothing
rewrites it on the way in, which is why the local model can feel worse
than MiniMax's demo reels. The demos are fed a structured prompt and
most people type a sentence. This is the single biggest thing you can
fix, and it costs nothing.

H3 wants named sections. For text to video, three:

```
integrated_multimodal_description: ...
overall_soundscape: ...
non_diegetic_music: ...
```

A thin overall_soundscape gives you a near-silent clip. The audio is
generated, not attached, so describe the ambience, the sounds of the
action, the breathing. non_diegetic_music wants instrumentation, tempo
and dynamics rather than mood words: "sparse fingerpicked guitar, slow,
dropping to near silence" works, "melancholy and hopeful" does not.

Aim for 350 to 500 words of description. MiniMax's own prompt-writing
guides cover shot-cut timestamps and dialogue blocks. If you would
rather not write all that, use one of the two Auto Prompt workflows and
give it a one-line idea instead.

## Settings you can change

Set these in the environment variables tab. Click Edit Template before
you deploy, or edit the variables on this pod and restart it.

| Variable | Default | What it does |
|---|---|---|
| download_minimax_h3 | unset | Set to true to download the models and install the workflows. Leave it unset and the pod boots with no models. |
| minimax_quant | int8 | Which build to download: int8, fp8, nvfp4, or false for full bf16. See below. |
| LLM_KEY | empty | Your OpenRouter key, for the Auto Prompt workflows. |

## Picking a quant

You can skip this. The default int8 runs natively on every GPU RunPod
rents and the workflows are already set up for it.

| minimax_quant | What you get |
|---|---|
| int8 (default) | The int8 model. Works well everywhere. |
| fp8 | The fp8 model. Native on 4090, L40, H100, H200 and RTX 50xx cards. Slower than int8 on older cards. |
| nvfp4 | Same files as fp8. NVFP4 only accelerates on Blackwell cards (RTX 50xx); other GPUs fall back. |
| false | The full bf16 FL2VA and Ref2VA models. Needs a much larger network volume. |

The text encoder is the same int8 build whichever quant you pick. Only
the quant you ask for is downloaded, and the workflows are pointed at
those files automatically, so you never touch a dropdown. If you set a
value that is not on the list, the pod tells you and uses int8.

The default quantized set needs at least a 90 GB network volume. The full
bf16 set needs at least 180 GB; 200 GB leaves comfortable room for outputs.

## Turbo LoRAs

The T2V, I2V and R2V workflows sample through a distilled turbo LoRA
from lightx2v. All six of their builds are downloaded, so trying
another one is a dropdown change in the Turbo LoRA node and nothing
else.

| LoRA in the dropdown | Steps | Strength | Sigma shift |
|---|---|---|---|
| minimax_h3_fl2v_lightx2v_turbo_4step_v0.1_comfy (the one loaded) | 4 | 0.5 | 12 video / 3 audio |
| minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16 | 4 | 1.0 | 6 video / 3 audio |
| minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16 | 8 | 1.0 | 12 video / 3 audio |
| minimax_h3_fl2v_turbo_8step_v1.0_768p_comfyui_bf16 | 8 | 1.0 | 6 video / 3 audio |
| minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16 | 4 | 1.0 | 12 video / 3 audio |
| minimax_h3_ref2v_turbo_8step_v1.0_768p_comfyui_bf16 | 8 | 1.0 | 6 video / 3 audio |

The FL2V v0.1 build has its strength baked into the weights, which is why
the node sits at 0.5. The other five ship their own alpha instead, so
start them at 1.0, and change the Beta scheduler's step count when you
switch.

The Ref2VA v0.1 build is the one distilled for the reference path, and it is
what MiniMax - R2V - Auto Prompt loads, at 0.85. The stock
video_minimax_h3_r2v graph does not: it is ComfyUI's own and has no
LoRA node in it. To use the file there, add a LoraLoaderModelOnly after
the ref2va model loader yourself.

## Latent upscaling

The MiniMax H3 Latent Upscaler node pack is installed on this template.
With download_minimax_h3 set to true, its 3D fp16 weight is downloaded
to models/latent_upscale_models. Add “Minimax H3 Latent Upscaler (3D)”
to your graph to use it. The six bundled workflows do not add it for you.

This upscales MiniMax's latent before decoding. It can shorten the
high-resolution part of a workflow, but it does not reduce peak VRAM.

## Spectrum

Spectrum is baked into the image. It skips some of H3's transformer
evaluations by forecasting the features they would have produced, which
makes sampling faster. Add the node "Spectrum Apply MiniMax H3", under
sampling/spectrum, to a graph to use it. Nothing uses it unless you add
it yourself.

The speedup is not free. Forecasting changes the denoising trajectory,
so the same prompt and seed will not give you the clip you get without
the node. A/B a seed before you commit to it, and turn it off for
finals.

## Dynamic VRAM is turned off here

This template starts ComfyUI with --disable-dynamic-vram. It is not a
performance choice. ComfyUI's dynamic VRAM mode has an open bug that hits
the MiniMax int8 models, and it shows up as a CUDA "illegal memory access"
that kills the run partway through. Turning it off avoids it.

You do not need to do anything. It is on your pod already.

If you want it back, set COMFY_EXTRA_ARGS to --enable-dynamic-vram and
restart the pod. Your setting is applied last, so it wins.

One thing worth knowing if you do hit that error: it breaks the whole pod,
not just the one job. Every later run fails the same way until ComfyUI is
restarted. Restart the pod and it clears.

## Live previews while sampling

The bundled workflows show an animated preview of the video while it is
sampling. That part has not changed.

For graphs you build from scratch, the animated preview now follows
VideoHelperSuite's own default, which is off. Earlier versions of this
template switched it on for you. To turn it on, open Settings, find the
VHS section, and enable animated previews.
