"""
Custom torch operations for graph instrumentation.

This module defines custom operations that can be used to instrument
torch graphs during compilation passes.
"""

import torch


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

