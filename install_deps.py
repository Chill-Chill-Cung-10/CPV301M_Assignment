#!/usr/bin/env python3
"""
Install/upgrade required dependencies for the face recognition pipeline.

Features:
- ArcFace embeddings (via ONNX Runtime)
- YuNet face detection (via OpenCV)
- PyTorch with CPU/CUDA support
"""

import subprocess
import sys


def install_packages():
    """Install required packages."""
    packages = [
        "opencv-python>=4.8.0",
        "numpy>=1.24",
        "onnxruntime>=1.16",
        "onnx>=1.14",
        "torch",
        "torchvision",
        "pillow>=9.0",
        "matplotlib>=3.6",
    ]
    
    print("Installing/updating dependencies...")
    for package in packages:
        print(f"\n  ▶ {package}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", package])
    
    print("\n✓ All dependencies installed successfully!")
    print("\nOptional: Install onnxruntime-gpu for GPU acceleration:")
    print("  pip install onnxruntime-gpu")


if __name__ == "__main__":
    try:
        install_packages()
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Installation failed: {e}", file=sys.stderr)
        sys.exit(1)

