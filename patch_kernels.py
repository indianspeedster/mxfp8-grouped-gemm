"""Monkey-patch torchao's MXFP8 kernels with our optimized grouped-gemms versions.

Usage: Run this BEFORE importing torchtitan or torchao.moe_training.
  python patch_kernels.py && torchrun ... torchtitan.train ...

Replaces:
  - triton_mxfp8_grouped_mm  → our optimized forward (1533 TFLOPS geomean, auto-tuned configs)
  - triton_mxfp8_wgrad       → our v2 forward-as-backward (1467 TFLOPS kernel-only)
"""

import sys
import os


def apply():
    """Import our kernels and swap them into torchao's rocm_mxfp8_mm."""
    # Add grouped-gemms to path
    grouped_gemms_dir = os.path.expanduser("~/shekhar/grouped-gemms")
    if grouped_gemms_dir not in sys.path:
        sys.path.insert(0, grouped_gemms_dir)

    from kernels.mxfp8.forward import triton_mxfp8_grouped_mm as our_forward
    from kernels.mxfp8.backward import triton_mxfp8_wgrad_v2 as our_wgrad

    import torchao.prototype.moe_training.kernels.mxfp8.rocm_mxfp8_mm as target

    target.triton_mxfp8_grouped_mm = our_forward
    target.triton_mxfp8_wgrad = our_wgrad

    # Also patch the v2 name (used by our modified dispatch)
    target.triton_mxfp8_wgrad_v2 = our_wgrad

    # Now patch the module's __dict__ so imports pick it up
    import torchao.prototype.moe_training.kernels.mxfp8 as pkg
    pkg.rocm_mxfp8_mm = target

    print("[patch_kernels] ✓ Replaced forward + backward with grouped-gemms optimized kernels")
    print(f"  Forward: {our_forward.__module__}")
    print(f"  Backward: {our_wgrad.__module__}")

    # Verify the patch took
    from torchao.prototype.moe_training.kernels.mxfp8.rocm_mxfp8_mm import (
        triton_mxfp8_grouped_mm,
        triton_mxfp8_wgrad,
    )
    assert triton_mxfp8_grouped_mm is our_forward, "Forward patch failed!"
    assert triton_mxfp8_wgrad is our_wgrad, "Backward patch failed!"
    print("[patch_kernels] ✓ Verified patch integrity")


if __name__ == "__main__":
    apply()
