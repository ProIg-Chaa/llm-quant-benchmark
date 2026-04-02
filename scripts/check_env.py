#!/usr/bin/env python3

import importlib
import os
import platform
import sys
from importlib import metadata
from pathlib import Path


def bootstrap_vendor_path() -> Path | None:
    project_root = Path(__file__).resolve().parents[1]
    vendor_path = os.environ.get("LLM_QUANT_VENDOR_PATH")
    candidate = Path(vendor_path) if vendor_path else project_root / ".vendor" / "stage3"
    if candidate.exists():
        sys.path.insert(0, str(candidate))
        return candidate
    return None


VENDOR_PATH = bootstrap_vendor_path()


PACKAGE_NAMES = [
    ("torch", "torch"),
    ("transformers", "transformers"),
    ("bitsandbytes", "bitsandbytes"),
    ("autoawq", "awq"),
    ("auto_gptq", "auto_gptq"),
    ("optimum", "optimum"),
    ("accelerate", "accelerate"),
    ("datasets", "datasets"),
    ("pandas", "pandas"),
]


def get_version(package_name: str) -> str:
    try:
        return metadata.version(package_name)
    except metadata.PackageNotFoundError:
        return "not_installed"


def main() -> int:
    print("== Runtime ==")
    print(f"python_executable: {sys.executable}")
    print(f"python_version: {sys.version.split()[0]}")
    print(f"platform: {platform.platform()}")
    print(f"cwd: {os.getcwd()}")
    print(f"cuda_visible_devices: {os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}")
    print(f"vendor_path: {VENDOR_PATH if VENDOR_PATH else '<none>'}")

    print("\n== Package Status ==")
    for package_name, import_name in PACKAGE_NAMES:
        try:
            importlib.import_module(import_name)
            status = "installed"
        except Exception as exc:
            status = f"missing ({type(exc).__name__}: {exc})"
        print(f"{package_name}: {status}, version={get_version(package_name)}")

    print("\n== Torch / CUDA ==")
    try:
        import torch
    except Exception as exc:
        print(f"torch_import_error: {type(exc).__name__}: {exc}")
        return 1

    print(f"torch_version: {torch.__version__}")
    print(f"torch_cuda_available: {torch.cuda.is_available()}")
    print(f"torch_cuda_version: {torch.version.cuda}")
    print(f"cudnn_available: {torch.backends.cudnn.is_available()}")
    print(f"cudnn_version: {torch.backends.cudnn.version()}")
    print(f"gpu_count: {torch.cuda.device_count()}")

    for idx in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(idx)
        total_mem_gb = props.total_memory / (1024 ** 3)
        print(
            f"gpu[{idx}]: name={props.name}, capability={props.major}.{props.minor}, "
            f"total_memory_gb={total_mem_gb:.2f}"
        )

    print("\n== Expected Project Status ==")
    print("required_for_stage1: torch, transformers")
    print("required_for_stage2: bitsandbytes")
    print("required_for_stage3: autoawq, auto_gptq, optimum")
    print("benchmark_backend: transformers")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
