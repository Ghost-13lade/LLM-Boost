"""
Search Tool for LLM Boost.
Provides DuckDuckGo search capabilities.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass

try:
    from duckduckgo_search import DDGS
    DDGS_AVAILABLE = True
except ImportError:
    DDGS_AVAILABLE = False


@dataclass
class SearchResult:
    """Represents a single search result."""
    title: str
    href: str
    body: str
    
    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary."""
        return {
            "title": self.title,
            "href": self.href,
            "body": self.body
        }
    
    def __str__(self) -> str:
        return f"**{self.title}**\n{self.body}\n[Source]({self.href})"


class SearchTool:
    """
    A tool for performing web searches using DuckDuckGo.
    Provides clean, formatted results for LLM consumption.
    """
    
    def __init__(
        self,
        max_results: int = 5,
        region: str = "us-en",
        safesearch: str = "moderate"
    ):
        """
        Initialize the search tool.
        
        Args:
            max_results: Maximum number of results to return
            region: Region for search results (e.g., 'us-en', 'uk-en')
            safesearch: SafeSearch setting ('on', 'moderate', 'off')
        """
        if not DDGS_AVAILABLE:
            raise ImportError(
                "duckduckgo-search is not installed. "
                "Run: pip install duckduckgo-search"
            )
        
        self.max_results = max_results
        self.region = region
        self.safesearch = safesearch
    
    def search(
        self,
        query: str,
        max_results: Optional[int] = None
    ) -> List[SearchResult]:
        """
        Perform a web search.
        
        Args:
            query: The search query
            max_results: Override max_results for this search
            
        Returns:
            List of SearchResult objects
        """
        max_results = max_results or self.max_results
        
        with DDGS() as ddgs:
            results = list(ddgs.text(
                keywords=query,
                region=self.region,
                safesearch=self.safesearch,
                max_results=max_results
            ))
        
        return [
            SearchResult(
                title=r.get("title", ""),
                href=r.get("href", ""),
                body=r.get("body", "")
            )
            for r in results
        ]
    
    def search_formatted(
        self,
        query: str,
        max_results: Optional[int] = None
    ) -> str:
        """
        Perform a search and return formatted results.
        
        Args:
            query: The search query
            max_results: Override max_results for this search
            
        Returns:
            Formatted string with search results
        """
        results = self.search(query, max_results)
        
        if not results:
            return "No results found."
        
        formatted = [f"Search results for: '{query}'\n"]
        for i, result in enumerate(results, 1):
            formatted.append(f"\n{i}. {result}\n")
        
        return "\n".join(formatted)
    
    def search_news(
        self,
        query: str,
        max_results: Optional[int] = None
    ) -> List[SearchResult]:
        """
        Perform a news search.
        
        Args:
            query: The search query
            max_results: Override max_results for this search
            
        Returns:
            List of SearchResult objects from news
        """
        max_results = max_results or self.max_results
        
        with DDGS() as ddgs:
            results = list(ddgs.news(
                keywords=query,
                region=self.region,
                safesearch=self.safesearch,
                max_results=max_results
            ))
        
        return [
            SearchResult(
                title=r.get("title", ""),
                href=r.get("url", r.get("href", "")),
                body=r.get("body", "")
            )
            for r in results
        ]
    
    def get_tool_definition(self) -> Dict[str, Any]:
        """
        Get the tool definition for LLM function calling.
        
        Returns:
            Dictionary with tool definition in OpenAI format
        """
        return {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "Search the web for current information. Use this when you need up-to-date information or facts.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query"
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "Maximum number of results to return (default: 5)",
                            "default": 5
                        }
                    },
                    "required": ["query"]
                }
            }
        }
    
    def execute(self, query: str, max_results: Optional[int] = None) -> str:
        """
        Execute the search tool (for function calling).
        
        Args:
            query: The search query
            max_results: Maximum number of results
            
        Returns:
            Formatted search results
        """
        return self.search_formatted(query, max_results)


# Convenience function for quick searches
def quick_search(query: str, max_results: int = 5) -> str:
    """
    Quick search function.
    
    Args:
        query: The search query
        max_results: Maximum number of results
        
    Returns:
        Formatted search results
    """
    tool = SearchTool(max_results=max_results)
    return tool.search_formatted(query)


def perform_search(query: str, max_results: int = 5) -> str:
    """
    Perform a web search and return formatted results.
    
    This is the main entry point for search functionality.
    Uses DuckDuckGo for privacy-focused searching.
    
    Args:
        query: The search query
        max_results: Maximum number of results to return (default: 5)
        
    Returns:
        Formatted string with search results, or "No results found." on failure.
    """
    if not query or not query.strip():
        return "No results found."
    
    try:
        tool = SearchTool(max_results=max_results)
        results = tool.search_formatted(query)
        return results
    except Exception as e:
        # Return user-friendly message on any failure
        return "No results found."
