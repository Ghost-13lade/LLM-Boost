"""
MLX Local Provider for LLM Boost.
Apple Silicon local inference using MLX framework.
"""

import os
from typing import List, Dict, Any, Optional, Generator, AsyncGenerator

from .base import BaseProvider, Message, ChatResponse

# Conditional import for MLX
try:
    from mlx_lm import load, generate
    from mlx_lm.sample_utils import make_sampler
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False


class MLXProvider(BaseProvider):
    """
    Provider for local LLM inference using Apple's MLX framework.
    Optimized for Apple Silicon (M1/M2/M3/M4) chips.
    
    Supports:
    - Custom model paths via MODEL_PATH env var
    - KV cache quantization via KV_BITS env var
    - Large context via MAX_CONTEXT env var
    """
    
    def __init__(
        self,
        model: Optional[str] = None,
        model_path: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs
    ):
        """
        Initialize the MLX provider.
        
        Args:
            model: Model name or HuggingFace model ID
            model_path: Local path to model (optional, can use MODEL_PATH env var)
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            **kwargs: Additional options (top_p, top_k, kv_bits, max_context, etc.)
        """
        if not MLX_AVAILABLE:
            raise ImportError(
                "MLX not found. Please install mlx-lm to use this provider.\n"
                "Run: pip install mlx-lm\n"
                "Note: This requires an Apple Silicon Mac (M1/M2/M3/M4)."
            )
        
        # Load model path from env var or parameter
        resolved_model = model or os.getenv("MODEL_PATH") or model_path
        
        if not resolved_model:
            raise ValueError(
                "Model path must be provided. Set MODEL_PATH environment variable "
                "or pass model/model_path parameter."
            )
        
        super().__init__(resolved_model, temperature, max_tokens, **kwargs)
        
        self.model_path = model_path or os.getenv("MODEL_PATH")
        self._model = None
        self._tokenizer = None
        
        # KV cache quantization settings from env
        self.kv_bits = kwargs.get("kv_bits") or int(os.getenv("KV_BITS", "0"))
        self.max_context = kwargs.get("max_context") or int(os.getenv("MAX_CONTEXT", "65536"))
        
        # Additional generation parameters
        self.top_p = kwargs.get("top_p", 0.9)
        self.top_k = kwargs.get("top_k", 50)
        self.repetition_penalty = kwargs.get("repetition_penalty", 1.0)
        
        # Lazy loading - model loaded on first use
        self._loaded = False
    
    def _load_model(self):
        """Load the model (lazy loading) with KV cache quantization."""
        if self._loaded:
            return
        
        model_path = self.model_path or self.model
        print(f"Loading MLX model from: {model_path}")
        
        # Build load kwargs
        load_kwargs = {}
        
        # KV cache quantization (if specified)
        if self.kv_bits > 0:
            print(f"  Using {self.kv_bits}-bit KV cache quantization")
            load_kwargs["kv_bits"] = self.kv_bits
        
        # Load model with settings
        try:
            self._model, self._tokenizer = load(model_path, **load_kwargs)
        except TypeError:
            # Fallback for older MLX versions that don't support kv_bits
            print("  Note: kv_bits may not be supported in this MLX version")
            self._model, self._tokenizer = load(model_path)
        
        self._loaded = True
        print("Model loaded successfully!")
    
    def generate(
        self,
        messages: List[Message],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Generate a response from the LLM.
        
        Args:
            messages: List of Message objects
            temperature: Optional temperature override
            max_tokens: Optional max tokens override
            
        Returns:
            Generated text string
        """
        response = self.chat(messages, temperature=temperature, max_tokens=max_tokens)
        return response.content
    
    def chat(
        self,
        messages: List[Message],
        **kwargs
    ) -> ChatResponse:
        """
        Send a chat completion request.
        
        Args:
            messages: List of Message objects
            **kwargs: Additional parameters (temperature, max_tokens, top_p, top_k)
            
        Returns:
            ChatResponse object
        """
        self._load_model()
        
        # Format messages for the model
        prompt = self._format_messages(messages)
        
        # Create sampler with generation parameters
        temp = kwargs.get("temperature", self.temperature)
        sampler = make_sampler(
            temp=temp,
            top_p=kwargs.get("top_p", self.top_p),
            top_k=kwargs.get("top_k", self.top_k)
        )
        
        # Generate response
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        
        response_text = generate(
            self._model,
            self._tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            sampler=sampler,
            verbose=False
        )
        
        return ChatResponse(
            content=response_text,
            model=self.model,
            usage=None,  # MLX doesn't provide token usage
            finish_reason="stop"
        )
    
    def stream(
        self,
        messages: List[Message],
        **kwargs
    ) -> Generator[str, None, None]:
        """
        Stream a chat completion response.
        
        Args:
            messages: List of Message objects
            **kwargs: Additional parameters
            
        Yields:
            Chunks of the response
        """
        self._load_model()
        
        prompt = self._format_messages(messages)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        temp = kwargs.get("temperature", self.temperature)
        
        sampler = make_sampler(
            temp=temp,
            top_p=kwargs.get("top_p", self.top_p),
            top_k=kwargs.get("top_k", self.top_k)
        )
        
        # MLX doesn't have native streaming, so we simulate it
        # by generating the full response and yielding chunks
        response = generate(
            self._model,
            self._tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            sampler=sampler,
            verbose=False
        )
        
        # Yield word by word for a streaming effect
        words = response.split()
        for i, word in enumerate(words):
            if i == 0:
                yield word
            else:
                yield " " + word
    
    async def achat(
        self,
        messages: List[Message],
        **kwargs
    ) -> ChatResponse:
        """
        Send an async chat completion request.
        
        Note: MLX doesn't have native async support,
        so this runs synchronously.
        
        Args:
            messages: List of Message objects
            **kwargs: Additional parameters
            
        Returns:
            ChatResponse object
        """
        return self.chat(messages, **kwargs)
    
    async def astream(
        self,
        messages: List[Message],
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """
        Stream an async chat completion response.
        
        Note: MLX doesn't have native async support,
        so this yields synchronously.
        
        Args:
            messages: List of Message objects
            **kwargs: Additional parameters
            
        Yields:
            Chunks of the response
        """
        for chunk in self.stream(messages, **kwargs):
            yield chunk
    
    def _format_messages(self, messages: List[Message]) -> str:
        """
        Format messages into a prompt string.
        
        Handles multiple chat template formats:
        - ChatML style (used by many models)
        - Llama-3 style
        - Generic fallback
        
        Args:
            messages: List of Message objects
            
        Returns:
            Formatted prompt string
        """
        # Check if tokenizer has a chat template
        if hasattr(self._tokenizer, 'apply_chat_template') and self._tokenizer:
            try:
                # Use the tokenizer's chat template
                formatted_messages = [
                    {"role": msg.role.value, "content": msg.content}
                    for msg in messages
                ]
                prompt = self._tokenizer.apply_chat_template(
                    formatted_messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                return prompt
            except Exception:
                pass  # Fall through to manual formatting
        
        # Manual formatting (ChatML style - widely compatible)
        formatted = []
        
        for msg in messages:
            role = msg.role.value
            content = msg.content
            
            if role == "system":
                formatted.append(f"<|im_start|>system\n{content}<|im_end|>\n")
            elif role == "user":
                formatted.append(f"<|im_start|>user\n{content}<|im_end|>\n")
            elif role == "assistant":
                formatted.append(f"<|im_start|>assistant\n{content}<|im_end|>\n")
        
        # Add assistant prefix for response
        formatted.append("<|im_start|>assistant\n")
        
        return "".join(formatted)
    
    @classmethod
    def is_available(cls) -> bool:
        """Check if MLX is available on this system."""
        return MLX_AVAILABLE
    
    @classmethod
    def get_recommended_models(cls) -> List[str]:
        """
        Get list of recommended models for MLX.
        
        Returns:
            List of recommended HuggingFace model IDs
        """
        return [
            "mlx-community/Mistral-7B-Instruct-v0.3-4bit",
            "mlx-community/Meta-Llama-3-8B-Instruct-4bit",
            "mlx-community/Phi-3-mini-4k-instruct-4bit",
            "mlx-community/Qwen2-7B-Instruct-4bit",
            "mlx-community/gemma-2-9b-it-4bit",
        ]