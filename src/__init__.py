"""
LLM Boost - A Python wrapper to maximize LLM intelligence.
"""

__version__ = "0.1.0"
__author__ = "LLM Boost Team"

# Core
from .core import (
    run_llm_boost,
    run_llm_boost_simple,
    run_llm_boost_with_history,
    load_prompt,
)

# Providers
from .providers import (
    BaseProvider,
    NetworkProvider,
    get_provider,
    create_provider,
    Message,
)

# Memory
from .memory import MemoryStorage, MemoryManager

# Tools
from .tools import SearchTool, perform_search

# Utils
from .utils import (
    OutputParser,
    transcribe_audio,
)

__all__ = [
    # Core
    "run_llm_boost",
    "run_llm_boost_simple",
    "run_llm_boost_with_history",
    "load_prompt",
    # Providers
    "BaseProvider",
    "NetworkProvider",
    "get_provider",
    "create_provider",
    "Message",
    # Memory
    "MemoryStorage",
    "MemoryManager",
    # Tools
    "SearchTool",
    "perform_search",
    # Utils
    "OutputParser",
    "transcribe_audio",
]