#!/usr/bin/env bash
# RefMod Studio folder loaders resolve these names under ComfyUI's input/.
# Select the available deployment root even without runtime environment vars.
# mkdir -p preserves uploaded references on repeated boots.
if test -d /workspace; then
    mkdir -p /workspace/ComfyUI/input/refmod_images \
             /workspace/ComfyUI/input/refmod_video
else
    mkdir -p /ComfyUI/input/refmod_images \
             /ComfyUI/input/refmod_video
fi
