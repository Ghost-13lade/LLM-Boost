"""
Network Provider for LLM Boost.
OpenAI and OpenAI-compatible API implementation.
Supports: OpenAI, LM Studio, Ollama, vLLM, and other compatible APIs.
"""

import os
from typing import List, Dict, Any, Optional, Generator, AsyncGenerator

from .base import BaseProvider, Message, ChatResponse

try:
    from openai import OpenAI, AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


class NetworkProvider(BaseProvider):
    """
    Provider for OpenAI-compatible APIs.
    Works with OpenAI, LM Studio, Ollama, vLLM, and other compatible services.
    """
    
    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs
    ):
        """
        Initialize the network provider.
        
        Loads configuration from environment variables:
        - API_KEY: API key for the service
        - API_BASE_URL: Base URL for the API endpoint
        - MODEL_PATH: Model name/identifier
        
        Args:
            model: Model identifier (defaults to MODEL_PATH env var or 'gpt-4')
            api_key: API key (defaults to API_KEY env var)
            base_url: Base URL for API (for LM Studio, Ollama, vLLM, etc.)
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            **kwargs: Additional options
        """
        if not OPENAI_AVAILABLE:
            raise ImportError(
                "openai is not installed. Run: pip install openai"
            )
        
        # Load from environment variables with fallbacks
        self.api_key = api_key or os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("API_BASE_URL") or os.getenv("LM_STUDIO_BASE_URL")
        
        # Model name can come from env var or parameter
        model_name = model or os.getenv("MODEL_PATH") or os.getenv("DEFAULT_MODEL", "gpt-4")
        
        super().__init__(model_name, temperature, max_tokens, **kwargs)
        
        # Allow local endpoints without API key (LM Studio, Ollama, vLLM)
        is_local_endpoint = self.base_url and (
            "localhost" in self.base_url or 
            "127.0.0.1" in self.base_url or
            "0.0.0.0" in self.base_url
        )
        
        if not self.api_key and not is_local_endpoint:
            raise ValueError(
                "API_KEY must be provided for remote endpoints. "
                "Set API_KEY environment variable or pass api_key parameter."
            )
        
        # Initialize OpenAI client
        client_kwargs = {}
        if self.api_key:
            client_kwargs["api_key"] = self.api_key
        if self.base_url:
            client_kwargs["base_url"] = self.base_url
        elif not self.api_key and is_local_endpoint:
            # Local endpoint without API key needs a dummy key for OpenAI SDK
            client_kwargs["api_key"] = "not-needed"
            if self.base_url:
                client_kwargs["base_url"] = self.base_url
        
        self.client = OpenAI(**client_kwargs)
        self.async_client = AsyncOpenAI(**client_kwargs)
    
    def generate(
        self,
        messages: List[Message],
        temperature: Optional[float] = None
    ) -> str:
        """
        Generate a response from the LLM.
        
        Args:
            messages: List of Message objects
            temperature: Optional temperature override
            
        Returns:
            Generated text string
        """
        response = self.chat(messages, temperature=temperature)
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
            **kwargs: Additional parameters (tools, tool_choice, etc.)
            
        Returns:
            ChatResponse object
        """
        formatted_messages = self._prepare_messages(messages)
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=formatted_messages,
            temperature=kwargs.get("temperature", self.temperature),
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
            **{k: v for k, v in kwargs.items() 
               if k not in ["temperature", "max_tokens"]}
        )
        
        choice = response.choices[0]
        
        return ChatResponse(
            content=choice.message.content or "",
            model=response.model,
            usage={
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            } if response.usage else None,
            finish_reason=choice.finish_reason
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
        formatted_messages = self._prepare_messages(messages)
        
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=formatted_messages,
            temperature=kwargs.get("temperature", self.temperature),
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
            stream=True,
            **{k: v for k, v in kwargs.items() 
               if k not in ["temperature", "max_tokens"]}
        )
        
        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    
    async def achat(
        self,
        messages: List[Message],
        **kwargs
    ) -> ChatResponse:
        """
        Send an async chat completion request.
        
        Args:
            messages: List of Message objects
            **kwargs: Additional parameters
            
        Returns:
            ChatResponse object
        """
        formatted_messages = self._prepare_messages(messages)
        
        response = await self.async_client.chat.completions.create(
            model=self.model,
            messages=formatted_messages,
            temperature=kwargs.get("temperature", self.temperature),
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
            **{k: v for k, v in kwargs.items() 
               if k not in ["temperature", "max_tokens"]}
        )
        
        choice = response.choices[0]
        
        return ChatResponse(
            content=choice.message.content or "",
            model=response.model,
            usage={
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            } if response.usage else None,
            finish_reason=choice.finish_reason
        )
    
    async def astream(
        self,
        messages: List[Message],
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """
        Stream an async chat completion response.
        
        Args:
            messages: List of Message objects
            **kwargs: Additional parameters
            
        Yields:
            Chunks of the response
        """
        formatted_messages = self._prepare_messages(messages)
        
        stream = await self.async_client.chat.completions.create(
            model=self.model,
            messages=formatted_messages,
            temperature=kwargs.get("temperature", self.temperature),
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
            stream=True,
            **{k: v for k, v in kwargs.items() 
               if k not in ["temperature", "max_tokens"]}
        )
        
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    
    def chat_with_tools(
        self,
        messages: List[Message],
        tools: List[Dict[str, Any]],
        tool_choice: str = "auto",
        **kwargs
    ) -> ChatResponse:
        """
        Send a chat completion request with tool calling.
        
        Args:
            messages: List of Message objects
            tools: List of tool definitions
            tool_choice: Tool choice strategy ('auto', 'none', or specific)
            **kwargs: Additional parameters
            
        Returns:
            ChatResponse object
        """
        return self.chat(
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            **kwargs
        )
    
    def get_available_models(self) -> List[str]:
        """
        Get list of available models.
        
        Returns:
            List of model identifiers
        """
        try:
            models = self.client.models.list()
            return [model.id for model in models.data]
        except Exception:
            return [self.model]