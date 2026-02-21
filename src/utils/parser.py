"""
Output Parser for LLM Boost.
Parses tagged outputs from LLM responses.
"""

import re
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field


@dataclass
class ParsedOutput:
    """Represents parsed output from an LLM response."""
    raw: str
    tags: Dict[str, str] = field(default_factory=dict)
    text: str = ""
    
    @property
    def thinking(self) -> Optional[str]:
        """Get the thinking content."""
        return self.tags.get("thinking")
    
    @property
    def analysis(self) -> Optional[str]:
        """Get the analysis content."""
        return self.tags.get("analysis")
    
    @property
    def answer(self) -> Optional[str]:
        """Get the answer content."""
        return self.tags.get("answer")
    
    def get_tag(self, tag_name: str) -> Optional[str]:
        """
        Get content of a specific tag.
        
        Args:
            tag_name: Name of the tag
            
        Returns:
            Content of the tag or None if not found
        """
        return self.tags.get(tag_name)
    
    def has_tag(self, tag_name: str) -> bool:
        """
        Check if a tag exists.
        
        Args:
            tag_name: Name of the tag
            
        Returns:
            True if tag exists, False otherwise
        """
        return tag_name in self.tags
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "raw": self.raw,
            "tags": self.tags,
            "text": self.text
        }


class OutputParser:
    """
    Parser for extracting tagged content from LLM outputs.
    
    Supports various tag formats:
    - XML-style: <tag>content</tag>
    - Bracket-style: [tag]content[/tag]
    """
    
    # Default tags to extract
    DEFAULT_TAGS = [
        "thinking",
        "analysis",
        "answer",
        "reasoning",
        "plan",
        "code",
        "result",
        "explanation",
        "summary",
        "action",
        "observation",
        "thought",
    ]
    
    def __init__(self, custom_tags: Optional[List[str]] = None):
        """
        Initialize the parser.
        
        Args:
            custom_tags: List of custom tags to extract
        """
        self.tags = custom_tags or self.DEFAULT_TAGS.copy()
        
        # Compile regex patterns for each tag
        self._patterns: Dict[str, re.Pattern] = {}
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compile regex patterns for all tags."""
        for tag in self.tags:
            # XML-style pattern: <tag>content</tag>
            pattern = re.compile(
                rf"<{tag}>(.*?)</{tag}>",
                re.DOTALL | re.IGNORECASE
            )
            self._patterns[tag] = pattern
    
    def add_tag(self, tag_name: str):
        """
        Add a tag to parse.
        
        Args:
            tag_name: Name of the tag to add
        """
        if tag_name not in self.tags:
            self.tags.append(tag_name)
            pattern = re.compile(
                rf"<{tag_name}>(.*?)</{tag_name}>",
                re.DOTALL | re.IGNORECASE
            )
            self._patterns[tag_name] = pattern
    
    def parse(self, text: str) -> ParsedOutput:
        """
        Parse text and extract tagged content.
        
        Args:
            text: The text to parse
            
        Returns:
            ParsedOutput object with extracted content
        """
        result = ParsedOutput(raw=text)
        
        for tag, pattern in self._patterns.items():
            match = pattern.search(text)
            if match:
                result.tags[tag] = match.group(1).strip()
        
        # Extract text not in any tag
        remaining = text
        for tag, pattern in self._patterns.items():
            remaining = pattern.sub("", remaining)
        
        # Clean up remaining text
        result.text = remaining.strip()
        
        return result
    
    def extract_tag(self, text: str, tag_name: str) -> Optional[str]:
        """
        Extract content of a specific tag from text.
        
        Args:
            text: The text to search
            tag_name: Name of the tag to extract
            
        Returns:
            Content of the tag or None if not found
        """
        pattern = re.compile(
            rf"<{tag_name}>(.*?)</{tag_name}>",
            re.DOTALL | re.IGNORECASE
        )
        match = pattern.search(text)
        return match.group(1).strip() if match else None
    
    def extract_all_tags(self, text: str) -> Dict[str, str]:
        """
        Extract all known tags from text.
        
        Args:
            text: The text to search
            
        Returns:
            Dictionary of tag names to content
        """
        result = {}
        for tag, pattern in self._patterns.items():
            match = pattern.search(text)
            if match:
                result[tag] = match.group(1).strip()
        return result
    
    def remove_tags(self, text: str) -> str:
        """
        Remove all tags from text, leaving only untagged content.
        
        Args:
            text: The text to process
            
        Returns:
            Text with all tags removed
        """
        result = text
        for tag, pattern in self._patterns.items():
            result = pattern.sub("", result)
        return result.strip()
    
    def replace_tag(
        self,
        text: str,
        tag_name: str,
        replacement: str
    ) -> str:
        """
        Replace content of a specific tag.
        
        Args:
            text: The text to process
            tag_name: Name of the tag to replace
            replacement: Replacement content
            
        Returns:
            Text with tag content replaced
        """
        pattern = re.compile(
            rf"<{tag_name}>.*?</{tag_name}>",
            re.DOTALL | re.IGNORECASE
        )
        return pattern.sub(f"<{tag_name}>{replacement}</{tag_name}>", text)


def parse_output(text: str, tags: Optional[List[str]] = None) -> ParsedOutput:
    """
    Convenience function to parse output.
    
    Args:
        text: The text to parse
        tags: Optional list of tags to extract
        
    Returns:
        ParsedOutput object
    """
    parser = OutputParser(custom_tags=tags)
    return parser.parse(text)


def extract_thinking(text: str) -> Optional[str]:
    """
    Extract thinking content from text.
    
    Args:
        text: The text to search
        
    Returns:
        Thinking content or None
    """
    return OutputParser().extract_tag(text, "thinking")


def extract_answer(text: str) -> Optional[str]:
    """
    Extract answer content from text.
    
    Args:
        text: The text to search
        
    Returns:
        Answer content or None
    """
    return OutputParser().extract_tag(text, "answer")