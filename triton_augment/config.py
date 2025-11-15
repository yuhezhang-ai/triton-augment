"""
Global configuration for Triton-Augment.

Author: yuhezhang-ai
"""

import os


# Auto-tuning configuration
# Set to True to enable kernel auto-tuning (tests multiple configs for optimal performance)
# Set to False to use fixed, sensible defaults (faster compilation, good-enough performance)
ENABLE_AUTOTUNE = os.getenv('TRITON_AUGMENT_ENABLE_AUTOTUNE', '0') == '1'


def enable_autotune():
    """
    Enable kernel auto-tuning for optimal performance.
    
    When enabled, Triton will test multiple kernel configurations
    and cache the best one for your GPU and image sizes.
    
    Example:
        ```python
        import triton_augment as ta
        ta.enable_autotune()
        # Now kernels will auto-tune on first use
        ```
    """
    global ENABLE_AUTOTUNE
    ENABLE_AUTOTUNE = True
    print("[Triton-Augment] Auto-tuning enabled. Kernels will auto-tune on first use.")


def disable_autotune():
    """
    Disable kernel auto-tuning and use fixed defaults.
    
    When disabled, kernels use fixed configurations that work well
    across most GPUs and image sizes without tuning overhead.
    
    Example:
        ```python
        import triton_augment as ta
        ta.disable_autotune()
        # Now kernels will use fixed defaults (faster, good performance)
        ```
    """
    global ENABLE_AUTOTUNE
    ENABLE_AUTOTUNE = False
    print("[Triton-Augment] Auto-tuning disabled. Using fixed kernel configurations.")


def is_autotune_enabled() -> bool:
    """
    Check if auto-tuning is currently enabled.
    
    Returns:
        bool: True if auto-tuning is enabled, False otherwise
    """
    return ENABLE_AUTOTUNE


__all__ = [
    'ENABLE_AUTOTUNE',
    'enable_autotune',
    'disable_autotune',
    'is_autotune_enabled',
]

