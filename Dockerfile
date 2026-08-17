# syntax=docker/dockerfile:1
# ============================================================================
# comfyui-minimax template image, built FROM the shared base
# (hearmeman/comfyui-base, comfyui-runtime/base/Dockerfile).
#
# The base owns: python 3.12 + /opt/venv (on PATH), the pinned torch trio +
# /torch-constraint.txt applied via ENV PIP_CONSTRAINT, pip tooling, pyyaml/
# gdown/triton/jupyterlab, huggingface_hub + hf_xet, opencv-python, ComfyUI
# pinned at COMFYUI_REF (v0.32.0, which ships the MiniMax-H3 core nodes and
# the comfy-kitchen int8 attention wiring) with /comfyui-approved-ref,
# ComfyUI-Manager, both SageAttention wheels under /opt/sage/, the CivitAI
# downloader, and ENV ORT_INDEX_ARGS (the per-CUDA-variant onnxruntime index
# nuance).
#
# This layer adds ONLY the minimax node set, the onnxruntime-gpu reassert, and
# the entrypoint. BASE_IMAGE is passed by CI from pins.json's "base_image";
# the default below mirrors that pin so a plain build stays coherent.
# ============================================================================
ARG BASE_IMAGE=hearmeman/comfyui-base:cu130-comfy0.32.0-torch2.11.0
FROM ${BASE_IMAGE}

# The minimax node set, culled 2026-08-13 to the packs the shipped workflows
# actually resolve nodes from. Five packs ARE load-bearing. Four for the
# bundled T2V/I2V workflows: KJNodes (ModelPreviewOverrideKJ, StringConstant,
# SomethingToString), rgthree (Power Lora Loader, Display Any),
# VideoHelperSuite (VHS_VideoCombine) and Openrouter_node (the OpenRouterNode
# inside the Auto Prompt subgraph; its API key is user-supplied via widget,
# LLM_KEY, or a JSON file in the node dir, never baked). The fifth is
# MiniMaxRefPack, ours, which supplies the single MiniMaxH3ReferencePack node
# in "MiniMax - R2V - Auto Prompt": a reference manager plus an OpenRouter
# call that writes the six-section Ref2VA prompt. Its only declared
# requirement is `requests`, and av / numpy / Pillow / requests are all in
# ComfyUI v0.32.0's own requirements.txt, so it adds no new dependency to
# this image. Its key resolution mirrors Openrouter_node's
# (widget -> OPENROUTER_API_KEY -> LLM_KEY), so the LLM_KEY a customer
# already sets for the Auto Prompt workflows covers this one too.
#
# The stock r2v workflow needs none of these: every node type in it resolves
# to ComfyUI core.
#
# ComfyUI-Spectrum-MiniMax-H3 is an optional accelerator, not load-bearing:
# it forecasts post-transformer features with Chebyshev ridge regression to
# skip transformer evaluations, and adds one node (Spectrum Apply MiniMax H3,
# under sampling/spectrum). Baked in because it is not registered in
# ComfyUI-Manager's default channel, which puts it on the "high risk" install
# path, and that path needs allow_git_url_install AND a loopback listener, so
# it is permanently unreachable from Manager on a pod (ComfyUI runs with
# --listen). It needs ComfyUI >= e377e263 (in v0.31.0) for the latent_shapes
# argument on outer_sample; the base's v0.32.0 pin satisfies that floor.
# Output is approximate by design, so it is off unless a graph adds the node.
#
# Cache-buster, on MiniMaxRefPack only. The four third-party packs do not
# version-gate a model release, which is why this image shipped without one.
# MiniMaxRefPack is different: it is ours, it is under active development, and
# a workflow in this repo is saved against its widget list. ComfyUI restores
# widgets POSITIONALLY, so a rebuild that silently reused a months-old cached
# clone would hand the shipped R2V workflow a node whose widget order predates
# the file, and the graph would come back with values in the wrong boxes. This
# ADD makes the layer track that pack's main. It is one RUN, so a push there
# re-clones all six at their current HEADs — that is the wan shape, and the
# packs are unpinned either way. Corollary, and it has bitten twice now (v6,
# v7): a tag cut purely to pick up a MiniMaxRefPack release lands on the same
# repo commit as the tag before it, so tag -> commit does NOT identify an
# image's node set. Read the pack HEADs off the build log, not off git.
ADD https://api.github.com/repos/Hearmeman24/ComfyUI-MiniMaxRefPack/git/refs/heads/main /tmp/minimax_refpack.ref
# PIP_CONSTRAINT (base-owned) applies to every requirements install below.
RUN for repo in \
    https://github.com/kijai/ComfyUI-KJNodes.git \
    https://github.com/rgthree/rgthree-comfy.git \
    https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git \
    https://github.com/gabe-init/ComfyUI-Openrouter_node.git \
    https://github.com/xmarre/ComfyUI-Spectrum-MiniMax-H3.git \
    https://github.com/Hearmeman24/ComfyUI-MiniMaxRefPack.git; \
    do \
        cd /ComfyUI/custom_nodes; \
        repo_dir=$(basename "$repo" .git); \
        git clone "$repo"; \
        if [ -f "/ComfyUI/custom_nodes/$repo_dir/requirements.txt" ]; then \
            pip install -r "/ComfyUI/custom_nodes/$repo_dir/requirements.txt"; \
        fi; \
        if [ -f "/ComfyUI/custom_nodes/$repo_dir/install.py" ]; then \
            python "/ComfyUI/custom_nodes/$repo_dir/install.py"; \
        fi; \
    done

# Force GPU onnxruntime. A node requirements file can pull in plain
# `onnxruntime` (CPU), which shadows the GPU install because both provide the
# same `onnxruntime` module and last install wins. This reassert therefore
# comes AFTER the clone loop, and no later RUN may pip install anything
# (comfyui-runtime base Dockerfile, onnxruntime ordering trap). ORT_INDEX_ARGS
# is base-owned data: the Azure onnxruntime-cuda-12 index on cu128, empty on
# cu130 where PyPI's onnxruntime-gpu links CUDA 13.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip uninstall -y onnxruntime onnxruntime-gpu 2>/dev/null || true; \
    pip install onnxruntime-gpu $ORT_INDEX_ARGS

# Build-time gate: the shipped image must expose the CUDA provider. Provider
# enumeration is import-only and works with no GPU present, so this fails the
# CI build, not a customer pod. CI greps this Dockerfile for the
# CUDAExecutionProvider assertion and for the no-pip-install-after-it rule.
RUN python3 -c "import onnxruntime; p = onnxruntime.get_available_providers(); assert 'CUDAExecutionProvider' in p, p; print('onnxruntime providers OK:', p)"

COPY src/start_script.sh /start_script.sh
RUN chmod +x /start_script.sh

CMD ["/start_script.sh"]
