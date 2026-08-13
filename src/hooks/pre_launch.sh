# shellcheck shell=bash
# pre_launch hook for comfyui-minimax (CONTRACTS.md section 7).
# No shebang on purpose: this file is SOURCED, never executed.
#
# SOURCED by /comfyui-runtime/src/start.sh immediately before the ComfyUI
# launch. Sourcing rules: no `exit`, no `set -e`, runs on every boot and must
# be idempotent, own errors handled here.
#
# Install comfy-aimdo + comfy-kitchen, deliberately unpinned so pods track
# the latest release (maintainer's choice; ported from the pre-migration
# src/start.sh:177-179,296-301). ComfyUI v0.32.0 wires H3's int8 attention
# through comfy-kitchen (pre-migration Dockerfile:33-40), so a failed install
# degrades that path but must never kill the boot: the old `exit 1` does NOT
# transfer (hooks are sourced; exiting here would kill start.sh). pip is a
# near no-op once both are installed, so re-runs on restarts are cheap.
echo "🔧 Installing comfy-aimdo + comfy-kitchen..."
if ! pip install comfy-aimdo comfy-kitchen > /tmp/pip_comfy_extras.log 2>&1; then
    echo "⚠️  comfy-aimdo/comfy-kitchen install failed (see /tmp/pip_comfy_extras.log)."
    echo "   H3's int8 attention path is degraded this boot. Restarting the pod retries the install."
    report_warn "comfy-aimdo/comfy-kitchen install failed; H3's int8 attention path is degraded this boot (ComfyUI wires it through comfy-kitchen). Restart the pod to retry the install."
else
    echo "✅ comfy-aimdo + comfy-kitchen installed"
fi
