#!/usr/bin/env bash
# One command from a full checkout: create the eval env if needed, validate, run.
set -euo pipefail
repro_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
repro_mode=evaluate
repro_config=""
export RED_LIBERO_ROOT="${RED_LIBERO_ROOT:-$repro_root/../red-libero}"
repro_args=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --help|-h)
            echo 'Usage: bash scripts/reproduce.sh --config FILE [--mode evaluate|attack|benign] [redvla options]'
            echo 'Set RED_LIBERO_ROOT to red-libero and start the model server selected by your config.'
            echo 'Default: Conda environment redvla; override its name with REDVLA_CONDA_ENV.'
            echo 'Use REDVLA_PYTHON=/existing/env/bin/python to reuse an environment without installing.'
            exit 0 ;;
        --config|--mode)
            [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; exit 2; }
            if [[ "$1" == --config ]]; then repro_config="$2"; else repro_mode="$2"; fi
            shift 2 ;;
        *) repro_args+=("$1"); shift ;;
    esac
done
case "$repro_mode" in evaluate|attack|benign) ;; *) echo 'Invalid --mode' >&2; exit 2 ;; esac
[[ -n "$repro_config" ]] || { echo '--config FILE is required; select a model evaluation configuration.' >&2; exit 2; }
export RED_LIBERO_ROOT="$(realpath -m -- "$RED_LIBERO_ROOT")"
# Resolve user-supplied paths before changing directories.
repro_config="$(realpath -m -- "$repro_config")"
if [[ -n "${REDVLA_PYTHON:-}" ]]; then
    repro_python="$(command -v "$REDVLA_PYTHON")" || { echo 'REDVLA_PYTHON was not found' >&2; exit 2; }
    repro_command=("$(realpath -ms -- "$repro_python")")
else
    source "$repro_root/scripts/conda_env.sh"
    redvla_find_conda
    if ! redvla_conda_python > /dev/null 2>&1 || ! "${repro_command[@]}" -c 'from importlib.metadata import version; version("redvla"); version("robosuite"); version("red-libero")' > /dev/null 2>&1; then
        bash "$repro_root/scripts/setup_env.sh"
    fi
fi
"${repro_command[@]}" - <<'PY'
from importlib.metadata import PackageNotFoundError, version
for name, expected in [('numpy', '1.26.4'), ('robosuite', '1.5.1')]:
    try:
        actual = version(name)
    except PackageNotFoundError:
        actual = 'not installed'
    if actual != expected:
        raise SystemExit(f'{name}=={expected} required; found {actual}. Select a compatible REDVLA_CONDA_ENV or REDVLA_PYTHON; see requirements/eval.txt.')
PY
export MUJOCO_GL="${MUJOCO_GL:-osmesa}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-$MUJOCO_GL}"
# Keep source checkout and example plugins ahead of unrelated installed copies.
export PYTHONPATH="$repro_root/src:$RED_LIBERO_ROOT:$repro_root${PYTHONPATH:+:$PYTHONPATH}"
cd "$repro_root"
repro_record="$repro_root/.local/reproduction/$(date -u +%Y%m%dT%H%M%SZ)-$$"
mkdir -p "$repro_record"
"${repro_command[@]}" -m pip freeze > "$repro_record/pip-freeze.txt"
"${repro_command[@]}" -m redvla doctor > "$repro_record/doctor.json"
"${repro_command[@]}" -m redvla "$repro_mode" --config "$repro_config" "${repro_args[@]}" --dry-run > "$repro_record/jobs.json"
echo "Resolved jobs and environment: $repro_record"
exec "${repro_command[@]}" -m redvla "$repro_mode" --config "$repro_config" "${repro_args[@]}"
