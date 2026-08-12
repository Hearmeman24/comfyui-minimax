#!/usr/bin/env python3
"""Self-check: the SageAttention source build must be paid once per volume.

The fallback build used to clone into /tmp and `pip install -e .`, both of
which live in the container's writable layer — so a restart recompiled from
scratch. src/sage_build.sh now caches the wheel on the network volume under a
key covering everything that changes the binary.

What must hold:
  * a cold cache clones and compiles, then leaves the wheel on the volume;
  * a warm cache installs that wheel and never touches git or the compiler;
  * a changed environment (torch, CUDA, python, GPU arch) rebuilds;
  * no volume means no cache, and the build still succeeds;
  * a build that produces no wheel fails loudly and writes no done marker.

Runs anywhere: git, pip and python3 are stubbed, so no GPU and no compiler.

Run: python3 tools/test_sage_cache.py
"""
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SAGE_BUILD = REPO / "src" / "sage_build.sh"
START = REPO / "src" / "start.sh"

WHEEL_NAME = "sageattention-2.2.0-cp312-cp312-linux_x86_64.whl"

# Stubs log every call to $STUB_LOG, one argv per line, so a test can assert on
# what the script actually ran. /bin/sh shebangs on purpose — a python3 shebang
# would resolve to the python3 stub sitting next to them.
STUBS = {
    "git": """#!/bin/sh
echo "git $*" >> "$STUB_LOG"
if [ "$1" = "clone" ]; then mkdir -p "$3/.git"; fi
exit 0
""",
    "pip": """#!/bin/sh
echo "pip $*" >> "$STUB_LOG"
if [ "$1" = "wheel" ] && [ -z "$STUB_PIP_WHEEL_EMPTY" ]; then
    for a in "$@"; do
        if [ -n "$next" ]; then mkdir -p "$a"; echo built > "$a/%s"; next=""; fi
        if [ "$a" = "--wheel-dir" ]; then next=1; fi
    done
fi
exit 0
""" % WHEEL_NAME,
    "python3": """#!/bin/sh
echo "python3 $*" >> "$STUB_LOG"
cat > /dev/null
echo "$STUB_CACHE_KEY"
exit 0
""",
}

failures = []


def check(cond, label):
    print(("  ok   " if cond else "  FAIL ") + label)
    if not cond:
        failures.append(label)


def run(cache_root, work_dir, bindir, key="k1", wheel_empty=False):
    """Run sage_build.sh against the stubs. Returns (returncode, calls)."""
    log = Path(work_dir) / "stub.log"
    log.write_text("")
    env = dict(os.environ)
    env.update(
        PATH=f"{bindir}:{env['PATH']}",
        STUB_LOG=str(log),
        STUB_CACHE_KEY=key,
        STUB_PIP_WHEEL_EMPTY="1" if wheel_empty else "",
    )
    proc = subprocess.run(
        ["bash", str(SAGE_BUILD), cache_root, work_dir],
        env=env,
        capture_output=True,
        text=True,
    )
    return proc, log.read_text().splitlines()


def make_stubs(root):
    bindir = Path(root) / "bin"
    bindir.mkdir()
    for name, body in STUBS.items():
        path = bindir / name
        path.write_text(body)
        path.chmod(0o755)
    return str(bindir)


def fresh(root, name):
    d = Path(root) / name
    d.mkdir()
    return str(d)


def main():
    print("syntax")
    for script in (SAGE_BUILD, START):
        rc = subprocess.run(["bash", "-n", str(script)]).returncode
        check(rc == 0, f"bash -n {script.relative_to(REPO)}")

    print("wiring")
    start = START.read_text()
    check(
        "bash /comfyui-minimax/src/sage_build.sh" in start,
        "start.sh runs the repo copy of sage_build.sh",
    )
    check(
        0 <= start.find('NETWORK_VOLUME="/workspace"') < start.find("sage_build.sh"),
        "NETWORK_VOLUME is resolved before the cache root is derived",
    )
    check("pip install -e ." not in start, "no editable install left in start.sh")

    with tempfile.TemporaryDirectory() as root:
        bindir = make_stubs(root)
        cache = fresh(root, "volume_cache")

        print("cold cache")
        work = fresh(root, "work1")
        proc, calls = run(cache, work, bindir)
        cached = Path(cache) / "k1" / WHEEL_NAME
        check(proc.returncode == 0, "exits 0")
        check(any(c.startswith("git clone") for c in calls), "clones the source")
        check(any(c.startswith("pip wheel") for c in calls), "compiles a wheel")
        check(
            any(f"pip install --no-deps --force-reinstall {work}" in c for c in calls),
            "installs the freshly built wheel",
        )
        check(cached.is_file(), "leaves the wheel in the cache")
        check(
            not list(Path(cache, "k1").glob(".*")),
            "leaves no temp file behind in the cache",
        )
        check(Path(work, "sage_build_done").is_file(), "writes the done marker")

        print("warm cache")
        work = fresh(root, "work2")
        proc, calls = run(cache, work, bindir)
        check(proc.returncode == 0, "exits 0")
        check(not any(c.startswith("git ") for c in calls), "never touches git")
        check(not any(c.startswith("pip wheel") for c in calls), "never recompiles")
        check(
            any(f"pip install --no-deps --force-reinstall {cached}" in c for c in calls),
            "installs the cached wheel",
        )
        check(Path(work, "sage_build_done").is_file(), "writes the done marker")

        print("changed environment")
        work = fresh(root, "work3")
        proc, calls = run(cache, work, bindir, key="k2")
        check(any(c.startswith("git clone") for c in calls), "rebuilds on a new key")
        check(
            (Path(cache) / "k2" / WHEEL_NAME).is_file() and cached.is_file(),
            "keeps both keys' wheels",
        )

        print("no network volume")
        work = fresh(root, "work4")
        proc, calls = run("", work, bindir)
        check(proc.returncode == 0, "exits 0")
        check(any(c.startswith("pip wheel") for c in calls), "still builds")
        check(Path(work, "sage_build_done").is_file(), "writes the done marker")

        print("build produces nothing")
        work = fresh(root, "work5")
        proc, calls = run(cache, work, bindir, key="k3", wheel_empty=True)
        check(proc.returncode != 0, "exits non-zero")
        check(not Path(work, "sage_build_done").is_file(), "writes no done marker")
        check(not Path(cache, "k3").exists(), "caches nothing")

    print()
    if failures:
        print(f"{len(failures)} check(s) failed")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    if not shutil.which("bash"):
        sys.exit("bash not found")
    sys.exit(main())
