"""
Glassbox - A library for instrumenting and inspecting torch graph compilation.
"""


def __getattr__(name):
    if name in ("PostAttentionInjector", "create_post_attention_injector", "custom_ops"):
        from .passes import (
            PostAttentionInjector,
            create_post_attention_injector,
            custom_ops,
        )
        _exports = {
            "PostAttentionInjector": PostAttentionInjector,
            "create_post_attention_injector": create_post_attention_injector,
            "custom_ops": custom_ops,
        }
        globals().update(_exports)
        return _exports[name]
    raise AttributeError(f"module 'glassbox' has no attribute {name!r}")


__all__ = ["PostAttentionInjector", "create_post_attention_injector"]
