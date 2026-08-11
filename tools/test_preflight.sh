#!/usr/bin/env bash
# Drives src/start_script.sh's network preflight with a stubbed curl, so the
# blocked-DNS path can be exercised without a pod. No GPU, no network.
set -u

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SCRIPT="$REPO_ROOT/src/start_script.sh"
STUBS="$(mktemp -d)"
trap 'rm -rf "$STUBS"' EXIT

fail=0
check() {
    if [ "$1" = "0" ]; then
        echo "PASS: $2"
    else
        echo "FAIL: $2"
        fail=1
    fi
}

# --- case 1: DNS blocked. curl exits 6 ("could not resolve host"), which is
# exactly what a pod with RunPod global networking enabled produces.
cat > "$STUBS/curl" <<'STUB'
#!/usr/bin/env bash
exit 6
STUB
cat > "$STUBS/git" <<'STUB'
#!/usr/bin/env bash
echo "STUB GIT REACHED: $*"
STUB
chmod +x "$STUBS/curl" "$STUBS/git"

out="$(PATH="$STUBS:$PATH" bash "$SCRIPT" 2>&1)"
rc=$?
echo "--- blocked-DNS run (exit $rc) ---"
echo "$out"
echo "----------------------------------"

if [ "$rc" -ne 0 ]; then check 0 "blocked DNS exits non-zero"; else check 1 "blocked DNS exits non-zero"; fi
grep -q "Global Networking" <<<"$out"; check $? "message names the RunPod setting"
grep -qi "could not resolve host" <<<"$out"; check $? "message owns the git error it replaces"
! grep -q "STUB GIT REACHED" <<<"$out"; check $? "repo sync never runs"

# --- case 2: network fine. The preflight stays silent and boot continues.
cat > "$STUBS/curl" <<'STUB'
#!/usr/bin/env bash
exit 0
STUB
chmod +x "$STUBS/curl"

out="$(PATH="$STUBS:$PATH" bash "$SCRIPT" 2>&1)"
grep -q "STUB GIT REACHED" <<<"$out"; check $? "healthy network reaches the repo sync"
! grep -q "Global Networking" <<<"$out"; check $? "healthy network prints no warning"

exit $fail
