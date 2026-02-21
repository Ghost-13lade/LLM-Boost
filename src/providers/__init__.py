"""
Providers module for LLM Boost.
Contains LLM backend implementations and a factory for easy switching.
"""

import os
from typing import Optional

from .base import BaseProvider, Message, ChatResponse, MessageRole
from .network import NetworkProvider

# MLX provider is imported conditionally
try:
    from .mlx_local import MLXProvider
    MLX_AVAILABLE = True
except ImportError:
    MLXProvider = None
    MLX_AVAILABLE = False


def get_provider(
    backend: Optional[str] = None,
    **kwargs
) -> BaseProvider:
    """
    Get a provider instance based on the backend type.
    
    This factory function makes it easy to switch between backends
    using the LLM_BACKEND environment variable.
    
    Args:
        backend: Backend type ('network' or 'mlx'). 
                 Defaults to LLM_BACKEND env var or 'network'.
        **kwargs: Additional arguments passed to the provider.
        
    Returns:
        Configured provider instance.
        
    Raises:
        ValueError: If backend is invalid or MLX is requested but not available.
    """
    # Determine backend
    backend = backend or os.getenv("LLM_BACKEND", "network").lower()
    
    if backend == "network":
        return NetworkProvider(**kwargs)
    
    elif backend == "mlx":
        if not MLX_AVAILABLE:
            raise ImportError(
                "MLX not found. Please install mlx-lm to use this provider.\n"
                "Run: pip install mlx-lm\n"
                "Note: This requires an Apple Silicon Mac (M1/M2/M3/M4)."
            )
        return MLXProvider(**kwargs)
    
    else:
        raise ValueError(
            f"Unknown backend: '{backend}'. "
            f"Supported backends: 'network', 'mlx'"
        )


def create_provider(
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    temperature: float = 0.7,
    **kwargs
) -> BaseProvider:
    """
    Create a provider with explicit parameters.
    
    This is an alternative to get_provider() for cases where you want
    to explicitly set all parameters rather than using environment variables.
    
    Args:
        model: Model name/path
        api_key: API key (for network provider)
        base_url: API base URL (for network provider)
        temperature: Sampling temperature
        **kwargs: Additional provider-specific arguments
        
    Returns:
        Configured provider instance.
    """
    backend = os.getenv("LLM_BACKEND", "network").lower()
    
    if backend == "network":
        return NetworkProvider(
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
            **kwargs
        )
    elif backend == "mlx":
        if not MLX_AVAILABLE:
            raise ImportError(
                "MLX not found. Please install mlx-lm to use this provider.\n"
                "Run: pip install mlx-lm"
            )
        return MLXProvider(
            model=model,
            temperature=temperature,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown backend: '{backend}'")


__all__ = [
    # Classes
    "BaseProvider",
    "NetworkProvider",
    "MLXProvider",
    "Message",
    "ChatResponse",
    "MessageRole",
    # Functions
    "get_provider",
    "create_provider",
    # Constants
    "MLX_AVAILABLE",
]