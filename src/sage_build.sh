#!/usr/bin/env bash
# Build SageAttention from source, once per environment.
#
# The old inline version cloned into /tmp and ran `pip install -e .`, so both
# artifacts — the source tree holding the compiled .so, and the editable install
# in /opt/venv — lived in the container's writable layer. RunPod recreates that
# layer on every restart, so the ~3-minute compile was paid on every boot of
# every pod whose GPU the baked wheel can't serve.
#
# The network volume is the only filesystem that survives a restart, and the
# wheel is the only artifact worth keeping. Build it once, cache it there under
# a key covering everything that changes the binary, and reinstall from the
# cache on later boots — seconds instead of minutes.
#
# Usage: sage_build.sh <cache-root|""> <work-dir>
#   cache-root  where built wheels are kept. Empty disables the cache, for pods
#               with no network volume, where nothing persists anyway.
#   work-dir    scratch for the clone, the fresh wheel, and the done marker.

set -uo pipefail

export SAGE_COMMIT=68de379
SAGE_REPO=https://github.com/thu-ml/SageAttention.git

CACHE_ROOT=${1:-}
WORK_DIR=${2:-/tmp}

export EXT_PARALLEL=4 NVCC_APPEND_FLAGS="--threads 8" MAX_JOBS=32

# The key has to cover everything that changes the compiled extension. GPU arch
# is in it because SageAttention's setup.py compiles only for the cards it can
# see at build time: a wheel built on an L40S carries no Hopper kernels, and
# installing it on an H200 would reproduce the exact fault this cache exists to
# stop paying for.
cache_key() {
    python3 - <<'PY'
import os
import sys

import torch

major, minor = torch.cuda.get_device_capability(0)
print("-".join([
    os.environ["SAGE_COMMIT"],
    "cp%d%d" % sys.version_info[:2],
    "torch" + torch.__version__.split("+")[0],
    "cu" + (torch.version.cuda or "none").replace(".", ""),
    "sm%d%d" % (major, minor),
]))
PY
}

install_wheel() {
    # --force-reinstall because the cu130 image already carries a sageattention
    # of this exact version from the baked wheel. Without it pip calls the
    # requirement satisfied and leaves the kernel-less build in place.
    pip install --no-deps --force-reinstall "$1"
}

KEY=$(cache_key 2>/dev/null)
CACHE_DIR=""
if [ -z "$KEY" ]; then
    echo "⚠️  Could not read torch/GPU details — building without the wheel cache."
elif [ -n "$CACHE_ROOT" ]; then
    CACHE_DIR="$CACHE_ROOT/$KEY"
fi

CACHED=""
[ -n "$CACHE_DIR" ] && CACHED=$(ls "$CACHE_DIR"/sageattention-*.whl 2>/dev/null | head -n1)

if [ -n "$CACHED" ]; then
    echo "♻️  Reusing cached SageAttention wheel: $CACHED"
    install_wheel "$CACHED" || exit 1
else
    SRC_DIR="$WORK_DIR/SageAttention"
    WHEEL_DIR="$WORK_DIR/sage_wheel"
    # A restart can leave a previous boot's clone behind; git refuses to clone
    # into a non-empty directory, so clear both before starting.
    rm -rf "$SRC_DIR" "$WHEEL_DIR"

    git clone "$SAGE_REPO" "$SRC_DIR" || exit 1
    git -C "$SRC_DIR" reset --hard "$SAGE_COMMIT" || exit 1

    # --no-build-isolation so the extension links against the pod's own torch.
    # The commit is pinned and ships no pyproject.toml, so there is no build
    # backend to fetch anyway.
    pip wheel --no-deps --no-build-isolation --wheel-dir "$WHEEL_DIR" "$SRC_DIR" || exit 1

    BUILT=$(ls "$WHEEL_DIR"/sageattention-*.whl 2>/dev/null | head -n1)
    if [ -z "$BUILT" ]; then
        echo "❌ pip wheel produced no sageattention wheel."
        exit 1
    fi
    install_wheel "$BUILT" || exit 1

    if [ -n "$CACHE_DIR" ]; then
        # Copy under a temp name, then rename. The rename is atomic within the
        # volume, so a pod dying mid-copy can't leave half a wheel for the next
        # boot to install.
        WHEEL_NAME=$(basename "$BUILT")
        TMP_WHEEL="$CACHE_DIR/.$WHEEL_NAME.$$"
        if mkdir -p "$CACHE_DIR" && cp "$BUILT" "$TMP_WHEEL" &&
            mv -f "$TMP_WHEEL" "$CACHE_DIR/$WHEEL_NAME"; then
            echo "💾 Cached SageAttention wheel at $CACHE_DIR/$WHEEL_NAME"
        else
            rm -f "$TMP_WHEEL"
            echo "⚠️  Could not write the wheel cache — the next boot will rebuild."
        fi
    fi
fi

echo "SageAttention build completed" > "$WORK_DIR/sage_build_done"
