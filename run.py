"""Entry point for the Traffic Monitoring System.

Usage:
    python run.py                   # default (python/ultralytics backend)
    python run.py --code python     # ultralytics PyTorch backend
    python run.py --code cpp        # ONNX Runtime C++ backend (faster, GPU)
"""
import os
import sys
import glob
import argparse


def _setup_nvidia_libs():
    """Add all nvidia CUDA 12 lib dirs to LD_LIBRARY_PATH for onnxruntime-gpu."""
    base = os.path.join(sys.prefix, "lib", f"python{sys.version_info.major}.{sys.version_info.minor}",
                        "site-packages", "nvidia")
    if not os.path.isdir(base):
        return
    dirs = []
    for entry in os.listdir(base):
        lib_dir = os.path.join(base, entry, "lib")
        if os.path.isdir(lib_dir):
            dirs.append(lib_dir)
    if dirs:
        os.environ["LD_LIBRARY_PATH"] = ":".join(dirs) + ":" + os.environ.get("LD_LIBRARY_PATH", "")


def main():
    parser = argparse.ArgumentParser(description="Traffic Monitoring System")
    parser.add_argument(
        "--code", choices=["python", "cpp"], default="python",
        help="Inference backend: 'python' (ultralytics) or 'cpp' (ONNX Runtime C++ GPU)"
    )
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()

    if args.code == "cpp":
        _setup_nvidia_libs()

    from app.core import config
    config.INFERENCE_BACKEND = args.code

    print(f"[CONFIG] Backend: {args.code}")
    print(f"[CONFIG] Server: http://{args.host}:{args.port}")

    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=args.host,
        port=args.port,
        reload=False,
        workers=1,
    )


if __name__ == "__main__":
    main()
