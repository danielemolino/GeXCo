"""GeXCo: geometry-aware conditioning for biplanar X-ray-to-CT reconstruction."""

__version__ = "1.0.0"

# Must precede the first CUDA context for deterministic CuBLAS inference.
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
