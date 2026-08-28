# shellcheck shell=bash
# pre_download hook for comfyui-minimax (CONTRACTS.md section 7).
# No shebang on purpose: this file is SOURCED, never executed.
#
# SOURCED by /comfyui-runtime/src/start.sh immediately before the provisioner,
# with TEMPLATE_DIR, COMFYUI_DIR, PERSIST_ROOT exported and report_warn in
# scope. Sourcing rules: no `exit`, no `set -e`, runs on EVERY boot and must
# be idempotent, own errors handled here (a nonzero return only warns).
#
# `minimax_precision` is intentionally absent/off by default. Unset or
# `convrot` preserves the existing minimax_quant selector; `bf16` overrides it
# with the internal bf16 swap profile so the provisioner queues the full FL2VA
# and Ref2VA files INSTEAD of a quantized pair. Normalize like the provisioner:
# surrounding whitespace and case do not change the choice, and a typo warns
# without turning a working pod modelless.
MINIMAX_PRECISION_VALUE="${minimax_precision:-}"
MINIMAX_PRECISION_VALUE="${MINIMAX_PRECISION_VALUE#"${MINIMAX_PRECISION_VALUE%%[![:space:]]*}"}"
MINIMAX_PRECISION_VALUE="${MINIMAX_PRECISION_VALUE%"${MINIMAX_PRECISION_VALUE##*[![:space:]]}"}"
MINIMAX_PRECISION_VALUE="$(printf '%s' "$MINIMAX_PRECISION_VALUE" | tr '[:upper:]' '[:lower:]')"
case "$MINIMAX_PRECISION_VALUE" in
    ""|convrot)
        ;;
    bf16)
        export minimax_quant=bf16
        echo "🎚️  minimax_precision=bf16: selecting the full bf16 FL2VA and Ref2VA models"
        ;;
    *)
        echo "⚠️  unknown minimax_precision=${minimax_precision@Q}; using convrot/current minimax_quant selection"
        report_warn "unknown minimax_precision; using convrot/current minimax_quant selection"
        ;;
esac
unset MINIMAX_PRECISION_VALUE
#
# OpenRouter volume-copy cleanup, ported from the pre-migration
# src/start.sh:160-175. ComfyUI-Openrouter_node (the OpenRouterNode inside
# the Auto Prompt subgraph) is baked into the image. Earlier tags cloned it
# onto the network volume instead, which extra_model_paths.yaml adds to
# ComfyUI's custom_nodes search path additively, so a volume carried over
# from one of those tags still holds a second copy, and two copies register
# the same NODE_CLASS_MAPPINGS twice. Drop the volume copy whenever the
# image ships the pack itself. Idempotent by construction: the rm arm needs
# both dirs to exist, and once the volume copy is gone it never re-fires.
OPENROUTER_DIR="$PERSIST_ROOT/custom_nodes/ComfyUI-Openrouter_node"
if [ -d "$COMFYUI_DIR/custom_nodes/ComfyUI-Openrouter_node" ]; then
    if [ -d "$OPENROUTER_DIR" ]; then
        echo "🧹 ComfyUI-Openrouter_node is baked into the image; removing the older network-volume copy"
        rm -rf "$OPENROUTER_DIR"
    fi
else
    echo "⚠️  ComfyUI-Openrouter_node is missing from the image; the Auto Prompt workflows will open with a red node."
    report_warn "ComfyUI-Openrouter_node is missing from the image; the Auto Prompt workflows will show a red node"
fi
