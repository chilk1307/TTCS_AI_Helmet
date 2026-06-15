#!/bin/bash
# Wrapper script to launch server with correct CUDA library paths

NVIDIA_BASE="$(python3 -c "import sys,os; print(os.path.join(sys.prefix,'lib',f'python{sys.version_info.major}.{sys.version_info.minor}','site-packages','nvidia'))")"

if [ -d "$NVIDIA_BASE" ]; then
    for d in "$NVIDIA_BASE"/*/lib; do
        [ -d "$d" ] && export LD_LIBRARY_PATH="$d:$LD_LIBRARY_PATH"
    done
fi

exec python run.py "$@"
