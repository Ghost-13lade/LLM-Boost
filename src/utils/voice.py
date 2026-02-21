"""
Voice utilities for LLM Boost.
Provides speech-to-text transcription capabilities.
"""

import io
from typing import Optional

# Try to import SpeechRecognition (primary, zero-dependency option)
try:
    import speech_recognition as sr
    SPEECH_RECOGNITION_AVAILABLE = True
except ImportError:
    SPEECH_RECOGNITION_AVAILABLE = False

# Try to import openai-whisper (optional, more accurate but heavier)
try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False


def transcribe_audio(
    audio_bytes: bytes,
    use_whisper: bool = False,
    whisper_model: str = "base",
    language: Optional[str] = None
) -> str:
    """
    Transcribe audio bytes to text.
    
    Uses SpeechRecognition (Google Free API) by default for a zero-dependency
    local experience. Optionally uses OpenAI Whisper for better accuracy.
    
    Args:
        audio_bytes: Raw audio data as bytes
        use_whisper: If True, use Whisper instead of SpeechRecognition
        whisper_model: Whisper model size ('tiny', 'base', 'small', 'medium', 'large')
        language: Language code (e.g., 'en-US', 'es-ES'). Auto-detected if None.
        
    Returns:
        Transcribed text string, or empty string on failure.
        
    Raises:
        ImportError: If no speech recognition library is available.
    """
    if not audio_bytes:
        return ""
    
    # Prefer Whisper if requested and available
    if use_whisper:
        if not WHISPER_AVAILABLE:
            raise ImportError(
                "Whisper is not installed. Install it with: pip install openai-whisper\n"
                "Alternatively, use SpeechRecognition (default) which doesn't require Whisper."
            )
        return _transcribe_with_whisper(audio_bytes, whisper_model, language)
    
    # Use SpeechRecognition (Google Free API)
    if not SPEECH_RECOGNITION_AVAILABLE:
        # Fall back to Whisper if SpeechRecognition is not available
        if WHISPER_AVAILABLE:
            return _transcribe_with_whisper(audio_bytes, whisper_model, language)
        raise ImportError(
            "No speech recognition library found.\n"
            "Install one of:\n"
            "  - SpeechRecognition: pip install SpeechRecognition (recommended, free)\n"
            "  - openai-whisper: pip install openai-whisper (more accurate, heavier)"
        )
    
    return _transcribe_with_speech_recognition(audio_bytes, language)


def _transcribe_with_speech_recognition(
    audio_bytes: bytes,
    language: Optional[str] = None
) -> str:
    """
    Transcribe audio using SpeechRecognition with Google's free API.
    
    Args:
        audio_bytes: Raw audio data as bytes
        language: Language code (e.g., 'en-US', 'es-ES')
        
    Returns:
        Transcribed text, or empty string on failure.
    """
    try:
        recognizer = sr.Recognizer()
        
        # Convert bytes to AudioFile
        audio_file = io.BytesIO(audio_bytes)
        
        with sr.AudioFile(audio_file) as source:
            # Record the audio data
            audio_data = recognizer.record(source)
        
        # Recognize speech using Google Speech Recognition (free)
        # Note: This uses Google's public API which has usage limits
        text = recognizer.recognize_google(
            audio_data,
            language=language if language else None
        )
        
        return text
        
    except sr.UnknownValueError:
        # Speech was unintelligible
        return ""
    except sr.RequestError as e:
        # API request failed
        print(f"Speech recognition service error: {e}")
        return ""
    except Exception as e:
        # Handle unsupported audio formats and other errors
        print(f"Transcription error: {e}")
        return ""


def _transcribe_with_whisper(
    audio_bytes: bytes,
    model_size: str = "base",
    language: Optional[str] = None
) -> str:
    """
    Transcribe audio using OpenAI Whisper.
    
    Args:
        audio_bytes: Raw audio data as bytes
        model_size: Whisper model size ('tiny', 'base', 'small', 'medium', 'large')
        language: Language code (e.g., 'en', 'es')
        
    Returns:
        Transcribed text, or empty string on failure.
    """
    try:
        # Load the Whisper model
        model = whisper.load_model(model_size)
        
        # Save bytes to a temporary file (Whisper needs a file path)
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            tmp_file.write(audio_bytes)
            tmp_path = tmp_file.name
        
        # Transcribe
        result = model.transcribe(
            tmp_path,
            language=language if language else None
        )
        
        # Clean up temp file
        import os
        os.unlink(tmp_path)
        
        return result.get("text", "").strip()
        
    except Exception as e:
        print(f"Whisper transcription error: {e}")
        return ""


def transcribe_file(
    file_path: str,
    use_whisper: bool = False,
    whisper_model: str = "base",
    language: Optional[str] = None
) -> str:
    """
    Transcribe audio from a file.
    
    Convenience function that reads a file and transcribes it.
    
    Args:
        file_path: Path to the audio file
        use_whisper: If True, use Whisper instead of SpeechRecognition
        whisper_model: Whisper model size
        language: Language code
        
    Returns:
        Transcribed text, or empty string on failure.
    """
    try:
        with open(file_path, "rb") as f:
            audio_bytes = f.read()
        return transcribe_audio(audio_bytes, use_whisper, whisper_model, language)
    except FileNotFoundError:
        print(f"Audio file not found: {file_path}")
        return ""
    except Exception as e:
        print(f"Error reading audio file: {e}")
        return ""


def is_speech_recognition_available() -> bool:
    """Check if SpeechRecognition is available."""
    return SPEECH_RECOGNITION_AVAILABLE


def is_whisper_available() -> bool:
    """Check if Whisper is available."""
    return WHISPER_AVAILABLE


def get_recommended_backend() -> str:
    """
    Get the recommended transcription backend.
    
    Returns:
        'speech_recognition', 'whisper', or 'none' if neither is available.
    """
    if SPEECH_RECOGNITION_AVAILABLE:
        return "speech_recognition"
    elif WHISPER_AVAILABLE:
        return "whisper"
    else:
        return "none"