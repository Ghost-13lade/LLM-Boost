"""
Base Provider for LLM Boost.
Abstract base class for all LLM providers.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Generator, AsyncGenerator
from dataclasses import dataclass
from enum import Enum


class MessageRole(Enum):
    """Enum for message roles."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    FUNCTION = "function"
    TOOL = "tool"


@dataclass
class Message:
    """Represents a chat message."""
    role: MessageRole
    content: str
    name: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format for API calls."""
        result = {
            "role": self.role.value,
            "content": self.content
        }
        if self.name:
            result["name"] = self.name
        return result
    
    @classmethod
    def system(cls, content: str) -> "Message":
        """Create a system message."""
        return cls(role=MessageRole.SYSTEM, content=content)
    
    @classmethod
    def user(cls, content: str) -> "Message":
        """Create a user message."""
        return cls(role=MessageRole.USER, content=content)
    
    @classmethod
    def assistant(cls, content: str) -> "Message":
        """Create an assistant message."""
        return cls(role=MessageRole.ASSISTANT, content=content)


@dataclass
class ChatResponse:
    """Represents a chat response."""
    content: str
    model: str
    usage: Optional[Dict[str, int]] = None
    finish_reason: Optional[str] = None
    
    def __str__(self) -> str:
        return self.content


class BaseProvider(ABC):
    """
    Abstract base class for LLM providers.
    All providers must implement these methods.
    """
    
    def __init__(
        self,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs
    ):
        """
        Initialize the provider.
        
        Args:
            model: The model identifier
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens in response
            **kwargs: Additional provider-specific options
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.options = kwargs
    
    @abstractmethod
    def generate(
        self,
        messages: List[Message],
        temperature: Optional[float] = None
    ) -> str:
        """
        Generate a response from the LLM.
        
        This is the simplified main method for generating responses.
        
        Args:
            messages: List of Message objects
            temperature: Optional temperature override
            
        Returns:
            Generated text string
        """
        pass
    
    @abstractmethod
    def chat(
        self,
        messages: List[Message],
        **kwargs
    ) -> ChatResponse:
        """
        Send a chat completion request.
        
        Args:
            messages: List of Message objects
            **kwargs: Additional parameters
            
        Returns:
            ChatResponse object
        """
        pass
    
    @abstractmethod
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
        pass
    
    @abstractmethod
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
        pass
    
    @abstractmethod
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
        pass
    
    def _prepare_messages(
        self,
        messages: List[Message]
    ) -> List[Dict[str, str]]:
        """
        Convert Message objects to API format.
        
        Args:
            messages: List of Message objects
            
        Returns:
            List of message dictionaries
        """
        return [msg.to_dict() for msg in messages]
    
    def count_tokens(self, text: str) -> int:
        """
        Estimate token count for a text.
        
        Args:
            text: The text to count
            
        Returns:
            Estimated token count
        """
        # Simple estimation: ~4 characters per token
        return len(text) // 4
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model='{self.model}')"