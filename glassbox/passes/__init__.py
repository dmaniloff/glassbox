"""
Custom compilation passes for graph instrumentation.
"""

from .injector import PostAttentionInjector, create_post_attention_injector

__all__ = ["PostAttentionInjector", "create_post_attention_injector"]

