#!/usr/bin/env bash
# Bootstrap only the evaluation environment; model environments remain separate.
set -euo pipefail
repro_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
repro_renderer="${MUJOCO_GL:-osmesa}"
export RED_LIBERO_ROOT="${RED_LIBERO_ROOT:-$repro_root/../red-libero}"
if [[ "${1:-}" == --help ]]; then
    echo 'Usage: REDVLA_CONDA_ENV=redvla bash scripts/setup_env.sh'
    echo 'RED_LIBERO_ROOT selects your red-libero checkout (default: ../red-libero).'
    echo 'Uses Anaconda/Miniconda: Python 3.10, NumPy 1.26.4, robosuite 1.5.1, CPU PyTorch 2.5.1.'
    echo 'Set CONDA_EXE for a custom Conda installation; MUJOCO_GL=egl selects GPU rendering.'
    exit 0
fi
[[ "$(uname -s)" == Linux ]] || { echo 'The simulator setup currently supports Linux.' >&2; exit 2; }
[[ -d "$repro_root/data/scenes" ]] || {
    echo 'Use the full source release containing data/scenes/.' >&2; exit 2;
}
[[ -f "$RED_LIBERO_ROOT/setup.py" && -d "$RED_LIBERO_ROOT/libero/libero/assets" ]] || {
    echo 'Set RED_LIBERO_ROOT to your red-libero checkout (default: ../red-libero).' >&2; exit 2;
}
export RED_LIBERO_ROOT="$(realpath -- "$RED_LIBERO_ROOT")"
source "$repro_root/scripts/conda_env.sh"
redvla_find_conda
repro_base="$("$repro_conda" info --base)"
# Check system graphics libraries before creating or modifying the environment.
"$repro_base/bin/python" - "$repro_renderer" <<'PY'
import ctypes
import ctypes.util
import sys
backend = sys.argv[1]
if backend not in ('osmesa', 'egl'):
    raise SystemExit('MUJOCO_GL must be osmesa or egl.')
name = 'OSMesa' if backend == 'osmesa' else 'EGL'
library = ctypes.util.find_library(name)
if not library:
    raise SystemExit(f'Missing lib{name}. See README system prerequisites before installing Python packages.')
ctypes.CDLL(library)
PY
if ! repro_python="$(redvla_conda_python 2>/dev/null)"; then
    "$repro_conda" env create --name "$repro_env" --file "$repro_root/environment.yml" --yes
    repro_python="$(redvla_conda_python)"
fi
"${repro_command[@]}" - <<'PY'
import sys
if sys.version_info[:2] != (3, 10):
    raise SystemExit('Evaluation setup requires Python 3.10. Choose a fresh REDVLA_CONDA_ENV name.')
PY
"${repro_command[@]}" -m pip install 'pip>=24' 'setuptools==80.9.0' 'wheel==0.45.1'
# CPU torch is sufficient for simulator-side state I/O and remote VLA evaluation.
"${repro_command[@]}" -m pip install 'torch==2.5.1' --index-url https://download.pytorch.org/whl/cpu
"${repro_command[@]}" -m pip install -c "$repro_root/requirements/eval.txt" -e "$repro_root[eval,dev]"
for legacy_package in libero redvla-libero; do
    if "${repro_command[@]}" -m pip show "$legacy_package" > /dev/null 2>&1; then
        "${repro_command[@]}" -m pip uninstall -y "$legacy_package"
    fi
done
"${repro_command[@]}" -m pip install -e "$RED_LIBERO_ROOT"
"${repro_command[@]}" -c 'from redvla.envs.backend import configure_red_libero; print(configure_red_libero())'
"${repro_command[@]}" -m pip check
"${repro_command[@]}" - <<'PY'
from importlib.metadata import version
for name, expected in [('numpy', '1.26.4'), ('robosuite', '1.5.1')]:
    actual = version(name)
    if actual != expected:
        raise SystemExit(f'{name}=={expected} required; found {actual}')
    print(f'{name}=={actual}')
PY
"${repro_command[@]}" "$repro_root/scripts/verify_data.py"
echo "Evaluation environment ready: $repro_python"
echo "Activate with: conda activate $repro_env"
