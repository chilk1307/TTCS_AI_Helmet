#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p results

NVIDIA_BASE="$(python3 -c "import sys,os; print(os.path.join(sys.prefix,'lib',f'python{sys.version_info.major}.{sys.version_info.minor}','site-packages','nvidia'))")"
if [ -d "$NVIDIA_BASE" ]; then
    for d in "$NVIDIA_BASE"/*/lib; do
        [ -d "$d" ] && export LD_LIBRARY_PATH="$d:${LD_LIBRARY_PATH:-}"
    done
fi

ORT_CAPI="$(python3 -c "import os, onnxruntime as ort; print(os.path.join(os.path.dirname(ort.__file__), 'capi'))")"
export LD_LIBRARY_PATH="$ORT_CAPI:${LD_LIBRARY_PATH:-}"

exec python3 benchmark.py "$@"
