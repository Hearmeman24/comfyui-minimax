#!/usr/bin/env bash
# RefMod Studio folder loaders resolve these names under ComfyUI's input/.
# mkdir -p preserves uploaded references on repeated boots.
mkdir -p "${PERSIST_ROOT:?}/input/refmod_images" \
         "$PERSIST_ROOT/input/refmod_video"
