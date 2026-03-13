"""
Glassbox - A library for instrumenting and inspecting torch graph compilation.
"""

from .passes import (  # noqa: F401
    PostAttentionInjector,
    create_post_attention_injector,
    custom_ops,
)

__all__ = ["PostAttentionInjector", "create_post_attention_injector"]
