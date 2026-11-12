#!/usr/bin/env bash
#SBATCH --job-name=pyqcd-sdist-final
#SBATCH --partition=gpu
# The site rejects --gpus=0. Reserve one card and hide it from Python.
#SBATCH --gpus=1
#SBATCH --cpus-per-task=20
#SBATCH --time=00:10:00
#SBATCH --output=/home/bingxing2/home/scx8080/PyQCD/data/runs/release-0.4.0-20261112/jobs/%j.sdist-final.out
#SBATCH --error=/home/bingxing2/home/scx8080/PyQCD/data/runs/release-0.4.0-20261112/jobs/%j.sdist-final.err

set -uo pipefail

source "$HOME/env.sh"

REPO=/home/bingxing2/home/scx8080/PyQCD
RUN="$REPO/data/runs/release-0.4.0-20261112"
JOB_ID="${SLURM_JOB_ID:-manual}"
STAGE="$REPO/build/release-0.4.0-$JOB_ID/source"
OUT="$REPO/build/release-0.4.0-$JOB_ID/dist"
VENV="$REPO/build/release-0.4.0-$JOB_ID/venv"
FALLBACK_VENV="$REPO/build/release-0.4.0-$JOB_ID/venv-python"
LOG="$RUN/jobs/$JOB_ID"

mkdir -p "$STAGE" "$OUT" "$LOG"

status=0
run_step() {
    local label=$1
    shift
    printf 'BEGIN %s\n' "$label"
    "$@" > "$LOG/$label.log" 2>&1
    local rc=$?
    printf 'END %s rc=%d\n' "$label" "$rc"
    if [ "$rc" -ne 0 ]; then
        status=$rc
    fi
    return "$rc"
}

export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export PYTHONUTF8=1
export CUDA_VISIBLE_DEVICES=""
export NVIDIA_VISIBLE_DEVICES=NoDevFiles

rsync -a \
    "$REPO/pyproject.toml" "$REPO/setup.py" "$REPO/MANIFEST.in" \
    "$REPO/LICENSE" "$REPO/README.md" "$REPO/README.zh-CN.md" \
    "$REPO/build-kupl.sh" "$REPO/build-nv.sh" "$REPO/env.sh" "$REPO/install.sh" \
    "$STAGE/"
rsync -a \
    --exclude='testing/' --exclude='__pycache__/' --exclude='*.so' \
    "$REPO/pyqcd" "$STAGE/"
rsync -a --exclude='build/' "$REPO/cpp" "$STAGE/"

cd "$STAGE"
run_step build_sdist env PYQCD_EINSUM_BACKEND=kupl \
    python -m build --sdist --outdir "$OUT" || true
SDIST="$OUT/pyqcd-0.4.0.tar.gz"

if [ "$status" -eq 0 ]; then
    run_step twine_check python -m twine check "$SDIST" || true
fi
if [ "$status" -eq 0 ]; then
    run_step content_audit python - "$SDIST" <<'PY'
import pathlib
import sys
import tarfile
from setuptools import find_packages

archive = pathlib.Path(sys.argv[1])
version = "0.4.0"
root = f"pyqcd-{version}"
with tarfile.open(archive) as tar:
    names = tar.getnames()

expected = find_packages(
    where=".", include=["pyqcd", "pyqcd.*"],
    exclude=["pyqcd.testing", "pyqcd.testing.*"],
)
missing = [
    package for package in expected
    if not any(name.startswith(f"{root}/{package.replace('.', '/')}/") for name in names)
]
assert not missing, f"missing packages: {missing}"
assert not any("pyqcd/testing/" in name for name in names), "pyqcd.testing leaked"
assert any(name.endswith("pyqcd/py.typed") for name in names), "py.typed missing"
assert any(name.endswith("README.zh-CN.md") for name in names), "zh README missing"
assert not any(name.endswith(".so") for name in names), "native binary leaked"
assert not any(name.endswith("AGENTS.md") for name in names), "AGENTS.md leaked"

members = [name for name in names if not name.endswith("/")]
print(f"sdist_ok version={version} packages={len(expected)} files={len(members)}")
PY
fi
if [ "$status" -eq 0 ]; then
    run_step sources_audit bash -c \
        "tar -xzOf '$SDIST' pyqcd-0.4.0/pyqcd.egg-info/SOURCES.txt | grep -c '/home/' >/dev/null && exit 1 || exit 0" \
        || true
fi
if [ "$status" -eq 0 ]; then
    run_step build_warning_audit bash -c \
        "! grep -iE 'deprecat|path must be relative' '$LOG/build_sdist.log'" \
        || true
fi

if [ "$status" -eq 0 ]; then
    run_step make_venv python -m venv "$VENV" || true
fi
if [ "$status" -eq 0 ]; then
    run_step install_sdist env \
        PYQCD_EINSUM_BACKEND=kupl \
        PYQCD_EINSUM_NATIVE_ARCH=0 \
        "$VENV/bin/pip" install --no-cache-dir \
        --index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple "$SDIST" || true
fi
if [ "$status" -eq 0 ]; then
    run_step verify_sdist_install bash -c "
        cd '$OUT' &&
        '$VENV/bin/python' - <<'PY'
import pathlib
import numpy as np
import pyqcd
import pyqcd.einsum as pe

assert pyqcd.__version__ == '0.4.0', pyqcd.__version__
assert 'site-packages' in pyqcd.__file__, pyqcd.__file__
package_dir = pathlib.Path(pyqcd.__file__).parent
assert (package_dir / 'py.typed').is_file(), 'installed py.typed missing'
info = pe.build_info()
assert info['backend'] in {'kupl', 'cuda'}, info
a = np.arange(6.0).reshape(2, 3)
b = np.arange(12.0).reshape(3, 4)
assert np.array_equal(pe.contract('ij,jk->ik', a, b), a @ b)
assert len(pe.specializations()) == 8638
print('installed', pyqcd.__version__, pyqcd.__file__)
print('backend', info['backend'], 'array_backend', info['array_backend'])
PY
    " || true
fi

if [ "$status" -eq 0 ]; then
    run_step make_fallback_venv python -m venv "$FALLBACK_VENV" || true
fi
if [ "$status" -eq 0 ]; then
    run_step install_fallback env PYQCD_SKIP_NATIVE_BUILD=1 \
        "$FALLBACK_VENV/bin/pip" install --no-cache-dir \
        --index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple "$SDIST" || true
fi
if [ "$status" -eq 0 ]; then
    run_step verify_fallback bash -c "
        cd '$OUT' &&
        '$FALLBACK_VENV/bin/python' - <<'PY'
import numpy as np
import pyqcd
import pyqcd.einsum as pe

assert pyqcd.__version__ == '0.4.0'
assert 'site-packages' in pyqcd.__file__
assert pe.build_info()['backend'] == 'python', pe.build_info()
a = np.arange(6.0).reshape(2, 3)
b = np.arange(12.0).reshape(3, 4)
assert np.array_equal(pe.contract('ij,jk->ik', a, b), a @ b)
print('fallback', pyqcd.__version__, pe.build_info()['backend'])
PY
    " || true
fi

{
    printf 'job_id=%s\n' "$JOB_ID"
    printf 'finished=%s\n' "$(date -Iseconds)"
    printf 'exit_code=%s\n' "$status"
    printf 'sdist=%s\n' "$SDIST"
    printf 'sdist_size=%s\n' "$(stat -c %s "$SDIST" 2>/dev/null || echo missing)"
    printf 'sdist_sha256=%s\n' "$(sha256sum "$SDIST" 2>/dev/null | awk '{print $1}')"
} > "$LOG/metadata.after.txt"

exit "$status"
