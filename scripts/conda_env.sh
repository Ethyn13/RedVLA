#!/usr/bin/env bash
# Shared Conda discovery for setup and reproduction; source from those scripts.
redvla_find_conda() {
    repro_conda="${CONDA_EXE:-}"
    if [[ -z "$repro_conda" ]]; then
        repro_conda="$(type -P conda || true)"
    fi
    if [[ -z "$repro_conda" ]]; then
        for candidate in "$HOME/anaconda3/bin/conda" "$HOME/miniconda3/bin/conda" /opt/conda/bin/conda; do
            if [[ -x "$candidate" ]]; then
                repro_conda="$candidate"
                break
            fi
        done
    fi
    repro_conda="$(command -v "$repro_conda")" || {
        echo 'Conda not found. Install Anaconda/Miniconda, or set CONDA_EXE=/path/to/bin/conda.' >&2
        return 2
    }
    repro_env="${REDVLA_CONDA_ENV:-redvla}"
    if [[ ! "$repro_env" =~ ^[a-zA-Z0-9_][a-zA-Z0-9_.-]*$ || "$repro_env" == base || "$repro_env" == root ]]; then
        echo 'REDVLA_CONDA_ENV must be a non-base environment name (default: redvla).' >&2
        return 2
    fi
    repro_command=("$repro_conda" run --no-capture-output -n "$repro_env" python)
}

redvla_conda_python() {
    "${repro_command[@]}" -c 'import sys; print(sys.executable)'
}
