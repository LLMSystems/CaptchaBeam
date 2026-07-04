"""Build and cache a dynamic-batch version of ddddocr's recognition model.

ddddocr ships ``common.onnx`` with the input batch axis pinned to 1, so it can
only score one image per ``session.run``. This utility rewrites that axis to be
symbolic (proven numerically identical to per-image inference) and caches the
result, enabling true batched inference for :class:`BatchedDdddOcrBackend`.
"""
from __future__ import annotations

import hashlib
from pathlib import Path


def _cache_dir() -> Path:
    d = Path.home() / ".cache" / "captchabeam"
    d.mkdir(parents=True, exist_ok=True)
    return d


def locate_ddddocr_ocr_model() -> Path:
    import ddddocr  # noqa: PLC0415

    # ddddocr's default recognition model (old=False, beta=False) is
    # common_old.onnx; common.onnx is only used with beta=True. The batched
    # backend uses the default ddddocr instance, so match common_old.onnx.
    path = Path(ddddocr.__file__).parent / "common_old.onnx"
    if not path.exists():
        raise FileNotFoundError(f"ddddocr recognition model not found at {path}")
    return path


def build_dynamic_model(src: Path, dst: Path) -> None:
    """Rewrite the input/output leading dim of ``src`` to symbolic and save to ``dst``."""
    try:
        import onnx  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "Batched inference needs onnx to re-export the model: pip install onnx"
        ) from exc

    model = onnx.load(str(src))
    for tensor in list(model.graph.input) + list(model.graph.output):
        dims = tensor.type.tensor_type.shape.dim
        if len(dims):
            dims[0].dim_param = "batch"
            dims[0].ClearField("dim_value")
    dst.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(dst))


def get_dynamic_model_path() -> Path:
    """Return the cached dynamic model, building it from ddddocr's on first use.

    Cache key includes the source size+mtime so a ddddocr upgrade rebuilds it.
    """
    src = locate_ddddocr_ocr_model()
    stat = src.stat()
    key = hashlib.sha1(f"{src}|{stat.st_size}|{int(stat.st_mtime)}".encode()).hexdigest()[:16]
    dst = _cache_dir() / f"common_dynamic_{key}.onnx"
    if not dst.exists():
        build_dynamic_model(src, dst)
    return dst
