"""Port MXFP8 ROCm: targeted patches to upstream pytorch repos."""

import shutil, os, filecmp

SRC_AO = os.path.expanduser("~/shekhar/ao")
SRC_TT = os.path.expanduser("~/shekhar/torchtitan")
DST_AO = os.path.expanduser("~/shekhar/ao_upstream")
DST_TT = os.path.expanduser("~/shekhar/torchtitan_upstream")

# ── torchao: copy ROCm kernel + modified Python files ──
print("=== Porting torchao ===")

FILES = [
    "kernels/mxfp8/rocm_mxfp8_mm.py",
    "kernels/mxfp8/__init__.py",
    "mxfp8_grouped_mm.py",
    "config.py",
    "tensor.py",
    "utils.py",
    "kernels/mxfp8/quant.py",
]

base = "torchao/prototype/moe_training"
for fname in FILES:
    src = f"{SRC_AO}/{base}/{fname}"
    dst = f"{DST_AO}/{base}/{fname}"
    if os.path.exists(src):
        shutil.copy2(src, dst)
        tag = "~" if os.path.exists(dst) else "+"
        print(f"  [{tag}] {fname}")

print("  ✓ torchao done\n")

# ── torchtitan: patch mx.py for ROCm ──
print("=== Porting torchtitan ===")

mx_path = f"{DST_TT}/torchtitan/components/quantization/mx.py"
with open(mx_path) as f:
    content = f.read()

# Patch 1: add is_ROCM import
content = content.replace(
    "from torchtitan.tools.utils import has_cuda_capability",
    "from torchtitan.tools.utils import has_cuda_capability\n\ntry:\n    from torchao.utils import is_ROCM\nexcept ImportError:\n    def is_ROCM():\n        return False"
)

# Patch 2: relax SM100 check to allow ROCm  (in MXFP8LinearConverter.__init__)
content = content.replace(
    'if not has_cuda_capability(10, 0):\n            raise ValueError("MXFP8 is only supported on SM100 or later architectures")',
    'if not has_cuda_capability(10, 0) and not is_ROCM():\n            raise ValueError("MXFP8 is only supported on SM100+ or gfx950+ architectures")'
)

# Patch 3: relax SM100 check in MXFP8GroupedExpertsConverter.__init__
content = content.replace(
    'if not has_cuda_capability(10, 0):\n            raise ValueError("MXFP8 is only supported on SM100 or later architectures")',
    'if not has_cuda_capability(10, 0) and not is_ROCM():\n            raise ValueError("MXFP8 is only supported on SM100+ or gfx950+ architectures")'
)

with open(mx_path, "w") as f:
    f.write(content)
print("  [+] mx.py: ROCm capability check added")


# ── torchtitan: add DSv3 MXFP8 config ──
cfg_path = f"{DST_TT}/torchtitan/models/deepseek_v3/config_registry.py"
with open(cfg_path) as f:
    cfg = f.read()

if "deepseek_v3_16b_mxfp8" not in cfg:
    # Ensure imports exist
    if "MXFP8GroupedExpertsConverter" not in cfg:
        cfg = cfg.replace(
            "from torchtitan.components.quantization.float8 import Float8Converter",
            "from torchtitan.components.quantization.float8 import Float8Converter\nfrom torchtitan.components.quantization.mx import MXFP8GroupedExpertsConverter"
        )
    if "ModelConvertersContainer" not in cfg:
        cfg = cfg.replace(
            "from torchtitan.protocols.model_converter import build_model_converters",
            "from torchtitan.protocols.model_converter import build_model_converters, ModelConvertersContainer"
        )

    new_cfg = '''

def deepseek_v3_16b_mxfp8() -> Trainer.Config:
    """DeepSeek V3 16B with MXFP8 expert quantization (ROCm)."""
    config = deepseek_v3_16b()
    config.model_converters = ModelConvertersContainer.Config(
        converters=[
            MXFP8GroupedExpertsConverter.Config(
                recipe_name="mxfp8_rceil",
                pad_multiple=32,
            ),
        ],
    )
    return config
'''
    cfg += new_cfg
    with open(cfg_path, "w") as f:
        f.write(cfg)
    print("  [+] config_registry.py: added deepseek_v3_16b_mxfp8")

# ── torchtitan: add grouped-gemms kernel monkey-patch ──
GG = os.path.expanduser("~/shekhar/grouped-gemms")
train_path = f"{DST_TT}/torchtitan/train.py"
with open(train_path) as f:
    tc = f.read()

if "GROUPED-GEMMS PATCH" not in tc:
    patch = '''
# GROUPED-GEMMS PATCH: override torchao MXFP8 kernels with optimized versions
import sys as _sys, functools as _ft
_gg_dir = "''' + GG + '''"
if _gg_dir not in _sys.path:
    _sys.path.insert(0, _gg_dir)
from kernels.mxfp8.forward import triton_mxfp8_grouped_mm as _gg_fwd_raw
from kernels.mxfp8.backward import triton_mxfp8_wgrad_v2 as _gg_bwd_raw
@_ft.wraps(_gg_fwd_raw)
def _gg_fwd(*a, **kw):
    kw.pop("ctas_per_cu", None)
    return _gg_fwd_raw(*a, **kw)
@_ft.wraps(_gg_bwd_raw)
def _gg_bwd(*a, **kw):
    kw.pop("ctas_per_cu", None)
    return _gg_bwd_raw(*a, **kw)
import torchao.prototype.moe_training.kernels.mxfp8.rocm_mxfp8_mm as _tgt
_tgt.triton_mxfp8_grouped_mm = _gg_fwd
_tgt.triton_mxfp8_wgrad = _gg_bwd
_tgt.triton_mxfp8_wgrad_v2 = _gg_bwd
del _sys, _ft, _gg_fwd_raw, _gg_bwd_raw, _gg_dir
# END GROUPED-GEMMS PATCH
'''
    tc = tc.replace("import os", patch + "\nimport os", 1)
    with open(train_path, "w") as f:
        f.write(tc)
    print("  [+] train.py: grouped-gemms kernel patch")

print("\n✓ All ports applied successfully!")
