"""Port MXFP8 ROCm support from indianspeedster forks to upstream pytorch repos."""

import shutil, os, filecmp

SRC_AO = os.path.expanduser("~/shekhar/ao")
SRC_TT = os.path.expanduser("~/shekhar/torchtitan")
DST_AO = os.path.expanduser("~/shekhar/ao_upstream")
DST_TT = os.path.expanduser("~/shekhar/torchtitan_upstream")
GG = os.path.expanduser("~/shekhar/grouped-gemms")

# ── 1. torchao: copy ROCm kernel + modified Python files ──
print("=== Porting torchao ===")

# Copy rocm_mxfp8_mm.py (new file for ROCm kernels)
src = f"{SRC_AO}/torchao/prototype/moe_training/kernels/mxfp8/rocm_mxfp8_mm.py"
dst = f"{DST_AO}/torchao/prototype/moe_training/kernels/mxfp8/rocm_mxfp8_mm.py"
shutil.copy2(src, dst)
print(f"  [+] {dst}")

# Copy modified __init__.py for kernels/mxfp8
src = f"{SRC_AO}/torchao/prototype/moe_training/kernels/mxfp8/__init__.py"
dst = f"{DST_AO}/torchao/prototype/moe_training/kernels/mxfp8/__init__.py"
if not filecmp.cmp(src, dst):
    shutil.copy2(src, dst)
    print(f"  [~] {dst}")

# Copy modified Python files that differ
for fname in ["mxfp8_grouped_mm.py", "config.py", "tensor.py", "utils.py"]:
    src = f"{SRC_AO}/torchao/prototype/moe_training/{fname}"
    dst = f"{DST_AO}/torchao/prototype/moe_training/{fname}"
    if os.path.exists(src) and not filecmp.cmp(src, dst):
        shutil.copy2(src, dst)
        print(f"  [~] {fname}")

# Copy mxfp8_linear.py if it differs
for fname in ["mxfp8_linear.py"]:
    src = f"{SRC_AO}/torchao/prototype/moe_training/{fname}"
    dst = f"{DST_AO}/torchao/prototype/moe_training/{fname}"
    if os.path.exists(src) and os.path.exists(dst) and not filecmp.cmp(src, dst):
        shutil.copy2(src, dst)
        print(f"  [~] {fname}")

# Copy quantization utils (triton_to_mxfp8_dim1 etc.)
for fname in ["quant.py"]:
    src = f"{SRC_AO}/torchao/prototype/moe_training/kernels/mxfp8/{fname}"
    dst = f"{DST_AO}/torchao/prototype/moe_training/kernels/mxfp8/{fname}"
    if os.path.exists(src) and os.path.exists(dst) and not filecmp.cmp(src, dst):
        shutil.copy2(src, dst)
        print(f"  [~] kernels/mxfp8/{fname}")

print("  ✓ torchao ported")


# ── 2. torchtitan: rewrite mx.py to support ROCm ──
print("\n=== Porting torchtitan ===")

mx_py = f"""{open(f'{DST_TT}/torchtitan/components/quantization/mx.py').read().split('# Copyright')[0]}# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field, fields
from importlib.util import find_spec
from typing import Literal

from torchtitan.components.quantization import QuantizationConverter
from torchtitan.models.common.moe import GroupedExperts
from torchtitan.models.common.nn_modules import Linear
from torchtitan.protocols.module import Module
from torchtitan.tools.logging import logger
from torchtitan.tools.utils import has_cuda_capability

from .utils import swap_token_dispatcher

try:
    from torchao.utils import is_ROCM
except ImportError:
    def is_ROCM():
        return False

try:
    from torchao.prototype.moe_training.mxfp8_linear import (
        MXFP8Linear as TorchAOMXFP8Linear,
    )

    class MXFP8Linear(TorchAOMXFP8Linear, Module):
        \"\"\"Inherits from Module (not Linear) to satisfy the Module protocol
        (init_states, _param_init) while avoiding MRO conflicts with
        Linear.__init__. Config still inherits from Linear.Config for
        field compatibility.
        \"\"\"

        @dataclass(kw_only=True, slots=True)
        class Config(Linear.Config):
            \"\"\"Drop-in replacement for Linear.Config that builds MXFP8Linear.\"\"\"

            pass

        def __init__(self, config: Config):
            TorchAOMXFP8Linear.__init__(
                self,
                config.in_features,
                config.out_features,
                bias=config.bias,
            )

except ImportError:
    MXFP8Linear = None


class MXFP8LinearConverter(QuantizationConverter):
    \"\"\"Replace matching Linear.Config with MXFP8Linear.Config.\"\"\"

    @dataclass(kw_only=True, slots=True)
    class Config(QuantizationConverter.Config):
        fqns: list[str] = field(default_factory=list)
        \"\"\"
        List of fully qualified names of modules to apply MXFP8 quantization to.
        Only Linear.Config entries whose FQN contains a match are converted.
        If empty, all Linear modules are converted.
        \"\"\"

    def __init__(self, config: Config):
        self.config = config

        if MXFP8Linear is None:
            raise ImportError(
                "torchao is not installed. Please install it to use MXFP8 linear layers."
            )

        # Allow ROCm (gfx950+) in addition to CUDA SM100+
        if not has_cuda_capability(10, 0) and not is_ROCM():
            raise ValueError("MXFP8 is only supported on SM100+ or gfx950+ architectures")

        if not self.config.model_compile_enabled:
            logger.warning(
                "torch.compile enablement is required for highest performance "
                "of MXFP8 dynamic quantization."
            )

    def convert(self, model_config) -> None:
        assert MXFP8Linear is not None
        fqns = self.config.fqns
        for fqn, config, parent, attr in model_config.traverse(Linear.Config):
            if not fqns or any(target_fqn in fqn for target_fqn in fqns):
                new_config = MXFP8Linear.Config(
                    in_features=config.in_features,
                    out_features=config.out_features,
                    bias=config.bias,
                    param_init=config.param_init,
                )
                if isinstance(parent, list):
                    parent[attr] = new_config
                else:
                    setattr(parent, attr, new_config)

        logger.info("Converted Linear layers to MXFP8Linear")


_mxfp8_experts_cache: dict[type, type] = {{}}


def _get_mxfp8_grouped_experts_cls(parent_cls: type) -> type:
    \"\"\"Get or create an MXFP8-quantized subclass of *parent_cls*.

    Works for any ``GroupedExperts`` subclass (e.g. gpt-oss variants).
    The returned class has a proper ``_owner`` set by ``__init_subclass__``.
    \"\"\"
    if parent_cls in _mxfp8_experts_cache:
        return _mxfp8_experts_cache[parent_cls]

    parent_config_cls = parent_cls.Config  # type: ignore[attr-defined]

    class MXFP8GroupedExperts(parent_cls):  # type: ignore[valid-type, misc]
        @dataclass(kw_only=True, slots=True)
        class Config(parent_config_cls):  # type: ignore[misc]
            recipe_name: str = "mxfp8_rceil"

        def __init__(self, config: Config):
            super().__init__(config)
            from torchao.prototype.moe_training.config import (
                MXFP8TrainingOpConfig,
                MXFP8TrainingRecipe,
            )
            from torchao.quantization.quant_api import quantize_

            recipe = MXFP8TrainingRecipe(config.recipe_name)
            mxfp8_op_config = MXFP8TrainingOpConfig.from_recipe(recipe)
            quantize_(
                self,
                config=mxfp8_op_config,
                filter_fn=lambda mod, _fqn: isinstance(mod, GroupedExperts),
            )

    MXFP8GroupedExperts.__name__ = f"MXFP8{{parent_cls.__name__}}"
    MXFP8GroupedExperts.__qualname__ = f"MXFP8{{parent_cls.__name__}}"
    _mxfp8_experts_cache[parent_cls] = MXFP8GroupedExperts
    return MXFP8GroupedExperts


class MXFP8GroupedExpertsConverter(QuantizationConverter):
    \"\"\"Apply MXFP8 quantization to MoE expert grouped GEMMs.\"\"\"

    @dataclass(kw_only=True, slots=True)
    class Config(QuantizationConverter.Config):
        recipe_name: Literal["mxfp8_rceil"] = "mxfp8_rceil"
        \"\"\"
        Quantization recipe name for grouped GEMMs. Options: ["mxfp8_rceil"]

        - mxfp8_rceil: MXFP8 dynamic quantization with RCEIL rounding mode
          when computing the e8m0 scale factors.
        \"\"\"
        pad_multiple: int = 32
        \"\"\"
        Pad per-expert token groups to this multiple for MXFP8 grouped GEMM alignment.
        The CuTeDSL quantization kernel on sm_100 requires multiples of 128.
        \"\"\"

    def __init__(self, config: Config):
        self.config = config

        if find_spec("torchao") is None:
            raise ImportError(
                "torchao is not installed. Please install it to use MXFP8 MoE training."
            )

        # Allow ROCm (gfx950+) in addition to CUDA SM100+
        if not has_cuda_capability(10, 0) and not is_ROCM():
            raise ValueError("MXFP8 is only supported on SM100+ or gfx950+ architectures")

        if not self.config.model_compile_enabled:
            logger.warning(
                "torch.compile enablement is required for highest performance "
                "of MXFP8 dynamic quantization."
            )

    def convert(self, model_config) -> None:
        for _fqn, config, parent, attr in model_config.traverse(GroupedExperts.Config):
            swap_token_dispatcher(config, self.config.pad_multiple)
            base_module_cls = type(config)._owner
            quantized_cls = _get_mxfp8_grouped_experts_cls(base_module_cls)
            config_cls = quantized_cls.Config  # type: ignore[attr-defined]
            new_config = config_cls(
                **{{f.name: getattr(config, f.name) for f in fields(config)}},
                recipe_name=self.config.recipe_name,
            )
            if isinstance(parent, list):
                parent[attr] = new_config
            else:
                setattr(parent, attr, new_config)

        logger.info(
            f"Converted GroupedExperts to use dynamic {{self.config.recipe_name}} "
            "quantization for grouped_mm ops"
        )
"""

# Write the patched mx.py
with open(f"{DST_TT}/torchtitan/components/quantization/mx.py", "w") as f:
    f.write(mx_py)
print("  [+] torchtitan/components/quantization/mx.py (ROCm support)")

# ── 3. Add DSv3 MXFP8 config to torchtitan config_registry ──
cfg_path = f"{DST_TT}/torchtitan/models/deepseek_v3/config_registry.py"
with open(cfg_path) as f:
    cfg_content = f.read()

if "deepseek_v3_16b_mxfp8" not in cfg_content:
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
    # Find the import for ModelConvertersContainer - if it doesn't exist, add it
    if "ModelConvertersContainer" not in cfg_content:
        import_line = "from torchtitan.components.quantization.float8 import Float8Converter"
        if import_line in cfg_content:
            cfg_content = cfg_content.replace(import_line, import_line + "\nfrom torchtitan.components.quantization.mx import MXFP8GroupedExpertsConverter")
        else:
            # Find a good import insertion point
            cfg_content = cfg_content.replace(
                "from torchtitan.components.quantization.float8 import Float8Converter",
                "from torchtitan.components.quantization.float8 import Float8Converter\nfrom torchtitan.components.quantization.mx import MXFP8GroupedExpertsConverter"
            )
    
    if "ModelConvertersContainer" not in cfg_content:
        cfg_content = cfg_content.replace(
            "from torchtitan.protocols.model_converter import build_model_converters",
            "from torchtitan.protocols.model_converter import build_model_converters, ModelConvertersContainer"
        )
    
    cfg_content += new_cfg
    
    with open(cfg_path, "w") as f:
        f.write(cfg_content)
    print("  [+] Added deepseek_v3_16b_mxfp8 config")
else:
    print("  [~] MXFP8 config already exists")

# ── 4. Add grouped-gemms kernel monkey-patch to train.py ──
train_path = f"{DST_TT}/torchtitan/train.py"
with open(train_path) as f:
    train_content = f.read()

if "ASH PATCH" not in train_content and "kernels.mxfp8" not in train_content:
    patch = '''
# ── GROUPED-GEMMS PATCH: override torchao MXFP8 kernels with optimized versions ──
import sys as _sys, functools as _ft
_gg_dir = "/home/amd/shekhar/grouped-gemms"
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
# ── END GROUPED-GEMMS PATCH ──
'''
    # Insert before imports
    if "import os" in train_content:
        train_content = train_content.replace("import os", patch + "\nimport os", 1)
    else:
        train_content = patch + "\n" + train_content
    
    with open(train_path, "w") as f:
        f.write(train_content)
    print("  [+] Added grouped-gemms kernel patch to train.py")
else:
    print("  [~] Kernel patch already in train.py")

print("\n✓ Port complete!")
