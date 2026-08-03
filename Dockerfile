# Use multi-stage build with caching optimizations
#
# CUDA variant is parametrized. Defaults are CUDA 13.0 + torch cu130 — the combo
# MiniMax-H3 was validated on (torch 2.11.0+cu130, H200, driver 580.159.04) and the
# one the baked SageAttention wheel links against. The `-cuda12` tag overrides these
# four ARGs back to CUDA 12.8 (see .circleci/config.yml):
#   CUDA_BASE_IMAGE / TORCH_PACKAGES / TORCH_INDEX_URL / CUDA_VARIANT
# cu130 is also REQUIRED for native NVFP4 (cuBLAS 13.x FP4 matmul); on cu128 cuBLAS
# returns NOT_SUPPORTED and ComfyUI falls back to fp16/fp8.
ARG CUDA_BASE_IMAGE=nvidia/cuda:13.0.3-cudnn-devel-ubuntu24.04
FROM ${CUDA_BASE_IMAGE} AS base

# Re-declare after FROM so the build stage can read them.
# The cu130 trio is pinned exactly: the SageAttention wheel baked below was built
# against torch 2.11.0+cu130, so bumping this pin means rebuilding that wheel.
# The `-cuda12` override uses the STABLE torch channel (not nightly): the nightly
# trio rotates daily and frequently skews (torch a day ahead of torchvision/
# torchaudio, which pin an exact older torch -> ResolutionImpossible). Stable wheels
# are released together, mutually coherent, and don't get garbage-collected —
# reproducible tag rebuilds.
ARG TORCH_PACKAGES="torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0"
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu130
ARG CUDA_VARIANT=cu130
ENV CUDA_VARIANT=${CUDA_VARIANT}

# ComfyUI is pinned to a release tag, not master. Native MiniMax-H3 support landed
# in 57500fc5 (PR #15224), an ancestor of v0.30.0 — below that tag the MiniMaxH3*
# nodes do not exist and all three bundled workflows open with red nodes. Pinning
# also means a `vN` git tag rebuilds to the same image months later, which is the
# same reason the cu128 variant uses the stable torch channel over nightly.
# Bump deliberately; the version assertion below is the floor, not the target.
ARG COMFYUI_REF=v0.30.1

# Consolidated environment variables
ENV DEBIAN_FRONTEND=noninteractive \
   PIP_PREFER_BINARY=1 \
   PYTHONUNBUFFERED=1 \
   CMAKE_BUILD_PARALLEL_LEVEL=8 \
   HF_XET_HIGH_PERFORMANCE=1

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.12 python3.12-venv python3.12-dev \
        python3-pip \
        curl ffmpeg ninja-build git aria2 git-lfs wget vim \
        libgl1 libglib2.0-0 build-essential gcc && \
    \
    # make Python3.12 the default python & pip
    ln -sf /usr/bin/python3.12 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip && \
    \
    python3.12 -m venv /opt/venv && \
    \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Use the virtual environment
ENV PATH="/opt/venv/bin:$PATH"

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install ${TORCH_PACKAGES} \
        --index-url ${TORCH_INDEX_URL}

# Freeze the torch family so the requirements.txt installs below cannot
# upgrade/downgrade it (which on cu130 would silently drop NVFP4 support).
RUN pip freeze | grep -E "^(torch|torchvision|torchaudio|torchsde)==" > /torch-constraint.txt

# Core Python tooling
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install packaging setuptools wheel

# Runtime libraries
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install pyyaml gdown triton jupyterlab jupyterlab-lsp \
        jupyter-server jupyter-server-terminals \
        ipykernel jupyterlab_code_formatter

# huggingface_hub (provides `hf` CLI + bundled hf_xet accelerator)
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade huggingface_hub

# ------------------------------------------------------------
# ComfyUI install — direct clone + pip install. Replaces comfy-cli,
# which used to clone the same repo and create a private .venv we then
# deleted anyway. Simpler, fewer indirection layers, no ~7 GB .venv.
#
# Tracking master would need a GitHub-API ADD to bust the layer cache every
# time master moved. Pinning to ${COMFYUI_REF} removes that whole problem: the
# ref is part of this RUN's command string, so bumping the ARG invalidates the
# layer on its own, and leaving it alone is a clean cache hit.
# ------------------------------------------------------------
RUN --mount=type=cache,target=/root/.cache/pip \
    git clone --depth=1 --branch ${COMFYUI_REF} \
        https://github.com/comfyanonymous/ComfyUI.git /ComfyUI \
    && pip install -r /ComfyUI/requirements.txt

# Hard floor, independent of the pin above: MiniMax-H3 lives in ComfyUI core
# (comfy_extras/nodes_minimax_h3.py), not in a custom node, and it landed in
# v0.30.0. This is what stops anyone overriding COMFYUI_REF to something older
# and shipping an image whose three bundled workflows open with red nodes.
RUN cd /ComfyUI \
    && test -f comfy_extras/nodes_minimax_h3.py \
    && python3 -c "from comfyui_version import __version__ as v; assert tuple(map(int, v.split('.')[:2])) >= (0, 30), v; print('ComfyUI', v, 'has MiniMax-H3')"

FROM base AS final
# Make sure to use the virtual environment here too
ENV PATH="/opt/venv/bin:$PATH"
# Needed twice over: a couple of ComfyUI's own comfy_extras modules
# (nodes_sdpose, sam3) import cv2, as do several of the custom node packs below.
# Without it they all land as IMPORT FAILED.
RUN pip install opencv-python

# Custom nodes. None of these are needed by the three bundled MiniMax-H3
# workflows — every node type in those resolves to ComfyUI core — so this is a
# deliberately curated general-purpose toolkit for people building their own
# graphs on top of H3, not a dependency set. Keep the list short: each pack is
# import surface that can break a boot.
#
# Install handling per pack, in this order: clone, then requirements.txt under
# the torch constraint (so nothing swaps out the pinned cu130 trio), then
# install.py if the pack ships one. UltimateSDUpscale needs --recursive for its
# submodule; Impact-Pack is the one that actually uses the install.py step.
RUN for repo in \
    https://github.com/kijai/ComfyUI-KJNodes.git \
    https://github.com/rgthree/rgthree-comfy.git \
    https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git \
    https://github.com/cubiq/ComfyUI_essentials.git \
    https://github.com/yolain/ComfyUI-Easy-Use.git \
    https://github.com/Fannovel16/ComfyUI-Frame-Interpolation.git \
    https://github.com/ssitu/ComfyUI_UltimateSDUpscale.git \
    https://github.com/ltdrdata/ComfyUI-Impact-Pack.git \
    https://github.com/1038lab/ComfyUI-RMBG.git \
    https://github.com/kijai/ComfyUI-segment-anything-2.git; \
    do \
        cd /ComfyUI/custom_nodes; \
        repo_dir=$(basename "$repo" .git); \
        if [ "$repo" = "https://github.com/ssitu/ComfyUI_UltimateSDUpscale.git" ]; then \
            git clone --recursive "$repo"; \
        else \
            git clone "$repo"; \
        fi; \
        if [ -f "/ComfyUI/custom_nodes/$repo_dir/requirements.txt" ]; then \
            pip install -r "/ComfyUI/custom_nodes/$repo_dir/requirements.txt" \
                --constraint /torch-constraint.txt; \
        fi; \
        if [ -f "/ComfyUI/custom_nodes/$repo_dir/install.py" ]; then \
            python "/ComfyUI/custom_nodes/$repo_dir/install.py"; \
        fi; \
    done

# Force GPU onnxruntime. Several of the packs above (RMBG, segment-anything-2,
# Impact-Pack) list plain `onnxruntime` (CPU) in their requirements, which
# shadows a GPU install because both packages provide the same `onnxruntime`
# Python module — last install wins. Reinstalling after the clone loop
# guarantees the image ships with the CUDA provider available.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip uninstall -y onnxruntime onnxruntime-gpu 2>/dev/null || true; \
    pip install onnxruntime-gpu

# ComfyUI-Manager. Cloned as lowercase `comfyui-manager` so it loads
# after any other custom_nodes (ComfyUI loads alphabetically — capital
# letters first), which is required for Manager to detect IMPORT FAILED
# states in earlier-loaded nodes.
RUN --mount=type=cache,target=/root/.cache/pip \
    git clone --depth=1 https://github.com/ltdrdata/ComfyUI-Manager.git \
        /ComfyUI/custom_nodes/comfyui-manager \
    && if [ -f /ComfyUI/custom_nodes/comfyui-manager/requirements.txt ]; then \
         pip install -r /ComfyUI/custom_nodes/comfyui-manager/requirements.txt \
             --constraint /torch-constraint.txt; \
       fi

# Bake the prebuilt SageAttention cu130 wheel. start.sh installs it at runtime ONLY
# on the cu130 variant (and only if a real kernel probe passes on the worker GPU);
# the `-cuda12` image carries the file but never uses it (the wheel links
# libcudart.so.13 and won't import on cu128). cp312 wheel matches this image's
# python3.12.
COPY sageattention-2.2.0-cp312-cp312-linux_x86_64.whl /opt/sage/

# cu130 build-time sanity check: fail fast if torch isn't CUDA 13 or sage can't load.
# Runs on the default build; no-op on the `-cuda12` override.
RUN if [ "$CUDA_VARIANT" = "cu130" ]; then \
        python3 -c "import torch; assert torch.version.cuda.startswith('13'), torch.version.cuda; print('torch', torch.__version__)" && \
        pip install --no-deps /opt/sage/sageattention-2.2.0-cp312-cp312-linux_x86_64.whl && \
        python3 -c "import sageattention; print('sageattention import OK')"; \
    fi

COPY src/start_script.sh /start_script.sh
RUN chmod +x /start_script.sh

CMD ["/start_script.sh"]
