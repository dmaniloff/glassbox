"""
Custom torch operations for graph instrumentation.

This module defines custom operations that can be used to instrument
torch graphs during compilation passes.
"""

from typing import Tuple

import torch


# Register a custom torch operation for capturing QKV values before attention
@torch.library.custom_op("glassbox::capture_qkv", mutates_args=())
def capture_qkv_op(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, layer_name: str
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Custom op to capture Q, K, V values before attention.

    This is a passthrough operation that logs statistics about the QKV tensors
    while preserving the original tensor data.

    Args:
        q: Query tensor
        k: Key tensor
        v: Value tensor
        layer_name: Name identifier for the layer being captured

    Returns:
        A tuple of clones of the input tensors (q, k, v)
    """
    q_mean = float(q.mean().item())
    k_mean = float(k.mean().item())
    v_mean = float(v.mean().item())
    print(
        f"[QKV_CAPTURE] {layer_name} - Q mean: {q_mean:.6f}, K mean: {k_mean:.6f}, V mean: {v_mean:.6f}"
    )
    return q.clone(), k.clone(), v.clone()


@capture_qkv_op.register_fake
def _(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, layer_name: str
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Fake implementation for torch.compile tracing.

    Output has same shapes/dtypes/devices as inputs.
    """
    return q.clone(), k.clone(), v.clone()


# Register a custom torch operation for capturing mean values
@torch.library.custom_op("glassbox::capture_mean", mutates_args=())
def capture_mean_op(x: torch.Tensor, layer_name: str) -> torch.Tensor:
    """
    Custom op to capture mean values.

    This is a passthrough operation that logs the mean of the tensor
    while preserving the original tensor data.

    Args:
        x: Input tensor
        layer_name: Name identifier for the layer being captured

    Returns:
        A clone of the input tensor
    """
    mean_val = float(x.mean().item())
    print(f"[MEAN_CAPTURE] {layer_name} attention mean: {mean_val:.6f}")
    return x.clone()  # Passthrough


# Register the fake/abstract implementation for torch.compile
@capture_mean_op.register_fake
def _(x: torch.Tensor, layer_name: str) -> torch.Tensor:
    """
    Fake implementation for torch.compile tracing.

    Output has same shape/dtype/device as input.
    """
    return x.clone()
