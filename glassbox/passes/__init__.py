"""
Custom compilation passes for graph instrumentation.
"""

from .injector import (
    BaseAttentionInjector,
    BeforeAttentionInjector,
    PostAttentionInjector,
    create_before_attention_injector,
    create_post_attention_injector,
)

__all__ = [
    "BaseAttentionInjector",
    "BeforeAttentionInjector",
    "PostAttentionInjector",
    "create_before_attention_injector",
    "create_post_attention_injector",
]
