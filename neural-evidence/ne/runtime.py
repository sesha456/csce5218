"""GPU detection and per-GPU batch/precision profiles.

Effective batch size is held at 32 on every GPU so results are comparable
across runtimes; only the per-device batch and gradient accumulation change.
"""
import torch

# (per-device batch, grad accumulation) for max_length 128 and 256
PROFILES = {
    "A100": {128: (32, 1), 256: (32, 1)},
    "L4":   {128: (32, 1), 256: (16, 2)},
    "T4":   {128: (16, 2), 256: (8, 4)},
    "CPU":  {128: (8, 4), 256: (4, 8)},
}


def gpu_kind() -> str:
    if not torch.cuda.is_available():
        return "CPU"
    name = torch.cuda.get_device_name(0)
    for k in ("A100", "H100", "L4", "T4"):
        if k in name:
            return "A100" if k == "H100" else k
    return "T4"  # conservative default for unknown GPUs


def precision() -> dict:
    """bf16 on Ampere+ (A100, L4), fp16 on T4, fp32 on CPU."""
    if not torch.cuda.is_available():
        return {"bf16": False, "fp16": False}
    # is_bf16_supported() is also True on T4 (emulated, slow), so check for Ampere+ hardware
    if torch.cuda.get_device_capability(0)[0] >= 8:
        return {"bf16": True, "fp16": False}
    return {"bf16": False, "fp16": True}


def batch_plan(max_length: int, effective: int = 32):
    per_dev, accum = PROFILES[gpu_kind()][128 if max_length <= 128 else 256]
    scale = max(1, effective // 32)
    return per_dev, accum * scale


def describe() -> str:
    kind = gpu_kind()
    if kind == "CPU":
        return "CPU only"
    p = torch.cuda.get_device_properties(0)
    return f"{p.name} ({p.total_memory/2**30:.0f} GB), profile {kind}, precision {precision()}"


def setup():
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    print(describe())
