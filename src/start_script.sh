#!/usr/bin/env bash
# Image entrypoint. On container restarts the writable layer persists,
# so `git clone` would fail silently (refuses to clone into existing
# dir) and we'd ship the stale start.sh from the first boot. Always
# fetch + hard-reset to origin/master to guarantee the runtime scripts
# reflect what's on the repo.
#
# The sync is wrapped in a retry loop: a transient DNS/network blip at
# boot used to abort the whole entrypoint (set -e) and kill the
# container ("Could not resolve host: github.com"). Now we retry with
# backoff, and if GitHub stays unreachable we fall back to whatever repo
# copy is already on disk rather than bricking the pod.
REPO_DIR=/comfyui-minimax
REPO_URL=https://github.com/Hearmeman24/comfyui-minimax.git

# Preflight: RunPod's Global Networking option leaves the pod without outbound
# DNS. The first thing to touch the network is the repo sync below, so the pod
# dies on "Could not resolve host: github.com" and the reader goes looking at
# GitHub instead of at the pod setting they need to change. Probe first and say
# what it actually is. One shot per host, short timeout: a pod with no network
# is not going to boot anyway, so fail fast and loud rather than slow and vague.
reachable() {
    curl -sS --head --max-time 10 "https://$1" > /dev/null 2>&1
}

if ! reachable github.com && ! reachable huggingface.co; then
    cat >&2 <<'EOF'

================================================================================
❌ THIS POD HAS NO OUTBOUND NETWORK

Neither github.com nor huggingface.co is reachable, so nothing this template
needs can be downloaded. Left alone this shows up further down as a git error
("Could not resolve host: github.com"), which is misleading. GitHub is fine.

The usual cause is Global Networking being enabled on the pod. With it on, the
pod gets no outbound DNS and the template cannot boot.

To fix it:
  1. Terminate this pod.
  2. Deploy the template again and turn Global Networking OFF in the pod
     configuration screen before you hit deploy.
  3. Start the pod. It will boot normally.

If Global Networking is already off, the data center may be having a network
outage. Try deploying in a different region.
================================================================================

EOF
    exit 1
fi

sync_repo() {
    if [ -d "$REPO_DIR/.git" ]; then
        git -C "$REPO_DIR" fetch --depth=1 origin master &&
        git -C "$REPO_DIR" reset --hard origin/master
    else
        rm -rf "$REPO_DIR" &&
        git clone --depth=1 "$REPO_URL" "$REPO_DIR"
    fi
}

ok=""
for attempt in 1 2 3 4 5; do
    if sync_repo; then ok=1; break; fi
    echo "⚠️  repo sync attempt $attempt failed (network/DNS?). Retrying in $((attempt * 5))s..."
    sleep $((attempt * 5))
done

if [ -z "$ok" ]; then
    if [ -d "$REPO_DIR/.git" ]; then
        echo "⚠️  GitHub unreachable after retries — booting with the existing on-disk repo copy (may be stale)."
    else
        echo "❌ Could not clone $REPO_URL after retries and no local copy exists. Aborting." >&2
        exit 1
    fi
fi

cp -f "$REPO_DIR/src/start.sh" /
cp -f "$REPO_DIR/src/hf_download_manager.py" /
cp -f "$REPO_DIR/src/workflow_provisioner.py" /
cp -f "$REPO_DIR/src/models_registry.json" /
bash /start.sh
