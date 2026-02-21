"""
Utils module for LLM Boost.
Provides utility functions and parsers.
"""

from .parser import OutputParser, ParsedOutput
from .voice import (
    transcribe_audio,
    transcribe_file,
    is_speech_recognition_available,
    is_whisper_available,
    get_recommended_backend,
)

__all__ = [
    # Parser
    "OutputParser",
    "ParsedOutput",
    # Voice
    "transcribe_audio",
    "transcribe_file",
    "is_speech_recognition_available",
    "is_whisper_available",
    "get_recommended_backend",
]
