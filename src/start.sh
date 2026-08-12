#!/usr/bin/env bash

# Use libtcmalloc for better memory management
TCMALLOC="$(ldconfig -p | grep -Po "libtcmalloc.so.\d" | head -n 1)"
export LD_PRELOAD="${TCMALLOC}"

# This is in case there's any special installs or overrides that needs to occur when starting the machine before starting ComfyUI
if [ -f "/workspace/additional_params.sh" ]; then
    chmod +x /workspace/additional_params.sh
    echo "Executing additional_params.sh..."
    /workspace/additional_params.sh
else
    echo "additional_params.sh not found in /workspace. Skipping..."
fi

# Set the network volume path. Resolved here, before the SageAttention block,
# because the source-build fallback caches its wheel on the volume and needs to
# know whether there is one.
NETWORK_VOLUME="/workspace"
URL="http://127.0.0.1:8188"
if [ ! -d "$NETWORK_VOLUME" ]; then
    echo "NETWORK_VOLUME directory '$NETWORK_VOLUME' does not exist. You are NOT using a network volume. Setting NETWORK_VOLUME to '/' (root directory)."
    NETWORK_VOLUME="/"
fi

# SageAttention strategy.
#   cu130 image: a prebuilt cu130 wheel is baked at /opt/sage. Install it and
#   verify with a REAL kernel launch on this worker's GPU — import success isn't
#   enough (an arch the wheel wasn't built for, e.g. B200/sm_100, imports but
#   can't launch). If the probe passes we skip the source build entirely.
#   Otherwise (cu128 image, an older image with no CUDA_VARIANT, or an
#   unsupported GPU) fall back to building SageAttention from source.
SAGE_FLAG=""
SAGE_PID=""

sage_kernel_probe() {
    python3 - >/dev/null 2>&1 <<'PROBE'
import torch
from sageattention import sageattn
q = torch.randn(1, 8, 128, 64, dtype=torch.float16, device="cuda")
sageattn(q, q.clone(), q.clone())
torch.cuda.synchronize()
PROBE
}

SAGE_WHEEL=$(ls /opt/sage/sageattention-*.whl 2>/dev/null | head -n1)
if [ "$CUDA_VARIANT" = "cu130" ] && [ -n "$SAGE_WHEEL" ]; then
    echo "Installing baked SageAttention wheel: $SAGE_WHEEL"
    pip install --no-deps "$SAGE_WHEEL" > /tmp/sage_wheel.log 2>&1
    if sage_kernel_probe; then
        echo "✅ SageAttention wheel kernel probe passed — skipping source build"
        SAGE_FLAG="--use-sage-attention"
    else
        echo "⚠️  SageAttention wheel probe failed on this GPU — falling back to source build"
    fi
fi

# Start SageAttention source build in the background (only if the wheel path
# didn't already give us a working kernel). sage_build.sh caches the wheel it
# produces on the network volume, so this is a ~3-minute compile on a volume's
# first boot and a few seconds on every boot after it. It runs from the repo
# copy the entrypoint already synced, same as the workflows below, so it reaches
# pods on a published tag without waiting on an image rebuild.
if [ -z "$SAGE_FLAG" ]; then
    echo "Starting SageAttention build..."
    # A previous boot's marker can still be here. Left in place, the wait loop
    # below sees it and returns instantly — probing sage while this build is
    # mid-reinstall. Clear it synchronously, before the build starts.
    rm -f /tmp/sage_build_done
    SAGE_CACHE_ROOT=""
    [ "$NETWORK_VOLUME" != "/" ] && SAGE_CACHE_ROOT="$NETWORK_VOLUME/.sage_wheel_cache"
    bash /comfyui-minimax/src/sage_build.sh "$SAGE_CACHE_ROOT" /tmp \
        > /tmp/sage_build.log 2>&1 &
    SAGE_PID=$!
    echo "SageAttention build started in background (PID: $SAGE_PID)"
fi

if [ "$NETWORK_VOLUME" = "/" ]; then
    echo "NETWORK_VOLUME directory doesn't exist. Starting JupyterLab on root directory..."
    jupyter-lab --ip=0.0.0.0 --allow-root --no-browser --NotebookApp.token='' --NotebookApp.password='' --ServerApp.allow_origin='*' --ServerApp.allow_credentials=True --notebook-dir=/ &
else
    echo "NETWORK_VOLUME directory exists. Starting JupyterLab..."
    jupyter-lab --ip=0.0.0.0 --allow-root --no-browser --NotebookApp.token='' --NotebookApp.password='' --ServerApp.allow_origin='*' --ServerApp.allow_credentials=True --notebook-dir=/workspace &
fi

# ComfyUI source stays in the image (ephemeral, fast local disk). Models,
# workflows, outputs, inputs, and user-added custom_nodes live on the
# network volume via extra_model_paths.yaml + --user/output/input-directory
# flags. This avoids the 5-minute mv of /ComfyUI to MooseFS on first boot.
COMFYUI_DIR="/ComfyUI"
PERSIST_ROOT="$NETWORK_VOLUME/ComfyUI"
WORKFLOW_DIR="$PERSIST_ROOT/user/default/workflows"
CUSTOM_NODES_DIR="$COMFYUI_DIR/custom_nodes"

mkdir -p "$PERSIST_ROOT/models" "$PERSIST_ROOT/user" \
         "$PERSIST_ROOT/output" "$PERSIST_ROOT/input" \
         "$PERSIST_ROOT/custom_nodes"

# Symlink user/output/input into /ComfyUI so ComfyUI uses its default
# code paths (passing --user-directory triggers a None-user_dir bug in
# user_manager.get_users on the current ComfyUI revision). Models +
# custom_nodes still go through extra_model_paths.yaml (below) because
# we want the *additive* behavior — image-baked custom_nodes + user
# additions — not a wholesale replacement.
if [ "$NETWORK_VOLUME" != "/" ]; then
    # First boot only: migrate baked user/ content (default templates,
    # schema) to the volume before swapping in the symlink. cp -an is
    # no-clobber, so re-runs on existing volumes are safe.
    if [ -d "$COMFYUI_DIR/user" ] && [ ! -L "$COMFYUI_DIR/user" ]; then
        cp -an "$COMFYUI_DIR/user/." "$PERSIST_ROOT/user/" 2>/dev/null || true
        rm -rf "$COMFYUI_DIR/user"
    fi
    for sub in user output input; do
        [ -L "$COMFYUI_DIR/$sub" ] || rm -rf "$COMFYUI_DIR/$sub" 2>/dev/null || true
        ln -sfn "$PERSIST_ROOT/$sub" "$COMFYUI_DIR/$sub"
    done
fi

# Generate extra_model_paths.yaml from the live $PERSIST_ROOT so paths
# always match the actual network volume (not a baked-in /workspace
# assumption). Skip the file + flag when there's no real persistent
# volume — PERSIST_ROOT would equal COMFYUI_DIR and ComfyUI's defaults
# already cover those paths.
EXTRA_PATHS_FLAG=""
if [ "$NETWORK_VOLUME" != "/" ]; then
    cat > "$COMFYUI_DIR/extra_model_paths.yaml" <<EOF
network_volume:
    base_path: $PERSIST_ROOT
    checkpoints: models/checkpoints
    clip: models/clip
    clip_vision: models/clip_vision
    controlnet: models/controlnet
    diffusion_models: models/diffusion_models
    embeddings: models/embeddings
    loras: models/loras
    style_models: models/style_models
    text_encoders: models/text_encoders
    unet: models/unet
    upscale_models: models/upscale_models
    latent_upscale_models: models/latent_upscale_models
    detection: models/detection
    vae: models/vae
    vae_approx: models/vae_approx
    custom_nodes: custom_nodes
EOF
    EXTRA_PATHS_FLAG="--extra-model-paths-config $COMFYUI_DIR/extra_model_paths.yaml"
else
    rm -f "$COMFYUI_DIR/extra_model_paths.yaml"
fi

echo "Downloading CivitAI download script to /usr/local/bin"
git clone "https://github.com/Hearmeman24/CivitAI_Downloader.git" || { echo "Git clone failed"; exit 1; }
mv CivitAI_Downloader/download_with_aria.py "/usr/local/bin/" || { echo "Move failed"; exit 1; }
chmod +x "/usr/local/bin/download_with_aria.py" || { echo "Chmod failed"; exit 1; }
rm -rf CivitAI_Downloader  # Clean up the cloned repo

# Custom nodes. The H3 nodes are ComfyUI core (>= v0.30.0); KJNodes, rgthree,
# VideoHelperSuite and Openrouter_node are baked into the image and cover the
# T2V/I2V workflows. Nothing is installed at boot any more.
#
# ComfyUI-Openrouter_node (the OpenRouterNode inside the Auto Prompt subgraph)
# is baked into the image. Earlier tags cloned it onto the network volume
# instead, which extra_model_paths.yaml adds to ComfyUI's custom_nodes search
# path additively — so a volume carried over from one of those tags still holds
# a second copy, and two copies register the same NODE_CLASS_MAPPINGS twice.
# Drop the volume copy whenever the image ships the pack itself.
OPENROUTER_DIR="$PERSIST_ROOT/custom_nodes/ComfyUI-Openrouter_node"
if [ -d "$COMFYUI_DIR/custom_nodes/ComfyUI-Openrouter_node" ]; then
    if [ -d "$OPENROUTER_DIR" ]; then
        echo "🧹 ComfyUI-Openrouter_node is baked into the image — removing the older network-volume copy"
        rm -rf "$OPENROUTER_DIR"
    fi
else
    echo "⚠️  ComfyUI-Openrouter_node is missing from the image — the Auto Prompt workflows will open with a red node."
fi

echo "🔧 Installing comfy-aimdo + comfy-kitchen..."
pip install comfy-aimdo comfy-kitchen &
COMFY_EXTRAS_PID=$!


export change_preview_method="true"


# ---------------------------------------------------------------
# Workflow-driven model provisioning. The provisioner walks the
# workflow folders for each enabled flag, resolves model basenames
# via models_registry.json, emits a manifest for hf_download_manager,
# and copies the matching workflow JSONs to $WORKFLOW_DIR.
#
# Recognized flags (env vars set to "true"):
#   download_minimax_h3
#
# minimax_quant picks which quant of the DiT + text encoder is pulled, and
# retargets the copied workflows onto it:
# The text encoder is int8_convrot in every case; the quant only picks the DiT.
#   int8  (default) int8_convrot DiT. The only quant with no emulated path on
#                   any card RunPod rents, Ampere included. Safe, not the
#                   fastest.
#   fp8             fp8_scaled DiT. Native on Ada/Hopper/Blackwell (L40S,
#                   H100, H200, RTX 50xx). Emulated on Ampere.
#   nvfp4           same files as fp8, kept so pods already setting it boot.
# ---------------------------------------------------------------
HF_QUEUE_FILE="/tmp/hf_download_queue.tsv"
PROVISIONER_FLAGS=()
if [ "$download_minimax_h3" = "true" ]; then
    PROVISIONER_FLAGS+=(--flag download_minimax_h3)
fi

if [ ${#PROVISIONER_FLAGS[@]} -eq 0 ]; then
    echo "ℹ️  download_minimax_h3 not enabled — skipping model phase."
    : > "$HF_QUEUE_FILE"
else
    python3 /workflow_provisioner.py \
        --registry /models_registry.json \
        --workflows-src /comfyui-minimax/workflows \
        --workflows-dst "$WORKFLOW_DIR" \
        --models-root "$NETWORK_VOLUME/ComfyUI/models" \
        --manifest "$HF_QUEUE_FILE" \
        --quant "${minimax_quant:-int8}" \
        "${PROVISIONER_FLAGS[@]}"

    echo "🔽 Starting HF download manager..."
    python3 /hf_download_manager.py "$HF_QUEUE_FILE"
fi

declare -A MODEL_CATEGORIES=(
    ["$NETWORK_VOLUME/ComfyUI/models/checkpoints"]="$CHECKPOINT_IDS_TO_DOWNLOAD"
    ["$NETWORK_VOLUME/ComfyUI/models/loras"]="$LORAS_IDS_TO_DOWNLOAD"
)

# Counter to track background jobs
download_count=0

# Ensure directories exist and schedule downloads in background
for TARGET_DIR in "${!MODEL_CATEGORIES[@]}"; do
    mkdir -p "$TARGET_DIR"
    MODEL_IDS_STRING="${MODEL_CATEGORIES[$TARGET_DIR]}"

    # Skip if the value is the default placeholder
    if [[ "$MODEL_IDS_STRING" == "replace_with_ids" ]]; then
        echo "⏭️  Skipping downloads for $TARGET_DIR (default value detected)"
        continue
    fi

    IFS=',' read -ra MODEL_IDS <<< "$MODEL_IDS_STRING"

    for MODEL_ID in "${MODEL_IDS[@]}"; do
        sleep 1
        echo "🚀 Scheduling download: $MODEL_ID to $TARGET_DIR"
        (cd "$TARGET_DIR" && download_with_aria.py -m "$MODEL_ID") &
        ((download_count++))
    done
done

echo "📋 Scheduled $download_count downloads in background"

# Wait for all downloads to complete
echo "⏳ Waiting for downloads to complete..."
while pgrep -x "aria2c" > /dev/null; do
    echo "🔽 LoRA Downloads still in progress..."
    sleep 5  # Check every 5 seconds
done


echo "✅ All models downloaded successfully!"

echo "All downloads completed!"


# Workflow copying is handled by the provisioner above (per enabled flag).
cd /

# VHS ships its animated latent preview off by default. The bundled workflows
# carry VHS_latentpreview themselves and drive the preview through KJNodes'
# ModelPreviewOverrideKJ + TAEH3, so this only sets the default for graphs
# people build from scratch.
#
# Manager's config.ini is no longer seeded here. Every value that block wrote
# already matched Manager's own defaults (security_level normal, network_mode
# public, db_mode cache, share_option all, component_policy workflow,
# update_policy stable-comfyui) apart from preview_method, which the in-graph
# preview override makes redundant.
if [ "$change_preview_method" == "true" ]; then
    echo "Enabling VHS animated latent previews by default..."
    sed -i '/id: *'"'"'VHS.LatentPreview'"'"'/,/defaultValue:/s/defaultValue: false/defaultValue: true/' $CUSTOM_NODES_DIR/ComfyUI-VideoHelperSuite/web/js/VHS.core.js
else
    echo "Skipping VHS latent preview default (change_preview_method is not 'true')."
fi

# Workspace as main working directory
echo "cd $NETWORK_VOLUME" >> ~/.bashrc


# Wait for the background pip install to complete; abort on failure.
if wait "$COMFY_EXTRAS_PID"; then
    echo "✅ comfy-aimdo+comfy-kitchen install complete"
else
    echo "❌ comfy-aimdo+comfy-kitchen install failed."
    exit 1
fi

echo "Renaming loras downloaded as zip files to safetensors files"
cd $LORAS_DIR
for file in *.zip; do
    mv "$file" "${file%.zip}.safetensors"
done

# Wait for the SageAttention source build to finish (only if we started one;
# the baked-wheel path leaves SAGE_PID empty and SAGE_FLAG already set).
if [ -n "$SAGE_PID" ]; then
    echo "Waiting for SageAttention build to complete..."
    while ps -p $SAGE_PID > /dev/null 2>&1 && ! [ -f /tmp/sage_build_done ]; do
        echo "⚙️  SageAttention build in progress, this may take up to 5 minutes."
        sleep 5
    done

    if [ -f /tmp/sage_build_done ] && sage_kernel_probe; then
        echo "✅ SageAttention build completed successfully!"
        SAGE_FLAG="--use-sage-attention"
    else
        echo "⚠️  SageAttention unavailable — starting ComfyUI without it. Check /tmp/sage_build.log"
    fi
fi

# Start ComfyUI

echo "▶️  Starting ComfyUI"

nohup python3 "$COMFYUI_DIR/main.py" --listen --enable-cors-header '*' $SAGE_FLAG \
    $EXTRA_PATHS_FLAG \
    > "$NETWORK_VOLUME/comfyui_${RUNPOD_POD_ID}_nohup.log" 2>&1 &

    # Counter for timeout
    counter=0
    max_wait=70

    until curl --silent --fail "$URL" --output /dev/null; do
        if [ $counter -ge $max_wait ]; then
            echo "⚠️  ComfyUI should be up by now. If it's not running, there's probably an error."
            echo ""
            echo "🛠️  Troubleshooting Tips:"
            if [ "$CUDA_VARIANT" = "cu130" ]; then
                echo "1. This is the experimental CUDA 13 image. Make sure your CUDA Version is set to 13.0+ in the additional filters tab before deploying."
                echo "2. If you are deploying using network storage, try deploying without it"
                echo "3. NVFP4 quants only accelerate on Blackwell GPUs (RTX 50xx / sm_120); other GPUs fall back to fp16/fp8."
            else
                echo "1. Make sure that your CUDA Version is set to 12.8/12.9 by selecting that in the additional filters tab before deploying the template"
                echo "2. If you are deploying using network storage, try deploying without it"
                echo "3. If you are using a B200 GPU, it is currently not supported (try the -cuda13 image tag)"
            fi
            echo "4. If all else fails, open the web terminal by clicking \"connect\", \"enable web terminal\" and running:"
            echo "   cat comfyui_${RUNPOD_POD_ID}_nohup.log"
            echo "   This should show a ComfyUI error. Please paste the error in HearmemanAI Discord Server for assistance."
            echo ""
            echo "📋 Startup logs location: $NETWORK_VOLUME/comfyui_${RUNPOD_POD_ID}_nohup.log"
            break
        fi

        echo "🔄  ComfyUI Starting Up... You can view the startup logs here: $NETWORK_VOLUME/comfyui_${RUNPOD_POD_ID}_nohup.log"
        sleep 2
        counter=$((counter + 2))
    done

    # Only show success message if curl succeeded
    if curl --silent --fail "$URL" --output /dev/null; then
        echo "🚀 ComfyUI is UP"
    fi

    sleep infinity
