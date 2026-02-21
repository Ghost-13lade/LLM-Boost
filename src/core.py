"""
Core Logic for LLM Boost.
The "Brain" that orchestrates all components.
"""

import os
import re
from typing import Optional, Tuple
from pathlib import Path

from .providers.base import BaseProvider, Message
from .memory import MemoryManager
from .tools import perform_search
from .tools import PythonREPL, run_python
from .utils import OutputParser


# Default paths
DEFAULT_PROMPTS_DIR = Path(__file__).parent.parent / "prompts"


def load_prompt(prompt_name: str, prompts_dir: Optional[Path] = None) -> str:
    """
    Load a prompt template from file.
    
    Args:
        prompt_name: Name of the prompt file (without .md extension)
        prompts_dir: Directory containing prompts (defaults to prompts/)
        
    Returns:
        Prompt template string
    """
    prompts_dir = prompts_dir or DEFAULT_PROMPTS_DIR
    prompt_path = prompts_dir / f"{prompt_name}.md"
    
    if prompt_path.exists():
        return prompt_path.read_text(encoding="utf-8")
    else:
        # Return a default prompt if file not found
        return "You are a helpful assistant. Please respond to the user's query."


def extract_confidence(text: str) -> Optional[int]:
    """
    Extract confidence score from response.
    
    Looks for patterns like:
    - <confidence>8/10</confidence>
    - Confidence: 8/10
    - [8/10]
    
    Args:
        text: Response text to parse
        
    Returns:
        Confidence score (0-10) or None if not found
    """
    patterns = [
        r"<confidence>(\d+)/10</confidence>",
        r"[Cc]onfidence[:\s]+(\d+)/10",
        r"\[(\d+)/10\]",
        r"confidence[:\s]*(\d+)(?:/10)?",
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            score = int(match.group(1))
            return min(max(score, 0), 10)  # Clamp to 0-10
    
    return None


def extract_python_code(text: str) -> Optional[str]:
    """
    Extract Python code from response.
    
    Looks for code in:
    - ```python ... ``` blocks
    - <python>...</python> tags
    - :code: ... :endcode: markers
    
    Args:
        text: Response text to parse
        
    Returns:
        Python code string or None if not found
    """
    # Check for ```python ... ``` blocks
    python_block = re.search(r'```python\s*\n(.*?)```', text, re.DOTALL)
    if python_block:
        return python_block.group(1).strip()
    
    # Check for <python>...</python> tags
    python_tag = re.search(r'<python>(.*?)</python>', text, re.DOTALL)
    if python_tag:
        return python_tag.group(1).strip()
    
    # Check for :code: ... :endcode: markers
    code_marker = re.search(r':code:\s*\n(.*?)(?=:endcode:|$)', text, re.DOTALL)
    if code_marker:
        return code_marker.group(1).strip()
    
    return None


def execute_python_code(code: str) -> str:
    """
    Execute Python code and return the result.
    
    Uses the sandboxed PythonREPL for safe execution.
    
    Args:
        code: Python code to execute
        
    Returns:
        Execution result string
    """
    try:
        return run_python(code)
    except Exception as e:
        return f"[Python Execution Error]: {str(e)}"


def build_messages(
    user_message: str,
    system_prompt: str,
    memory_context: str = "",
    search_results: str = "",
    conversation_history: str = ""
) -> list:
    """
    Build the messages list for the LLM.
    
    Args:
        user_message: The user's input
        system_prompt: The system prompt template
        memory_context: Context from memory (RAG)
        search_results: Search results (Web)
        conversation_history: Recent conversation context
        
    Returns:
        List of Message objects
    """
    # Inject context into system prompt
    enhanced_prompt = system_prompt
    
    # Replace NEW placeholders (God Mode prompt)
    if "{web_context}" in enhanced_prompt:
        if search_results:
            enhanced_prompt = enhanced_prompt.replace(
                "{web_context}",
                search_results
            )
        else:
            enhanced_prompt = enhanced_prompt.replace("{web_context}", "No web search results available.")
    
    if "{rag_context}" in enhanced_prompt:
        if memory_context:
            enhanced_prompt = enhanced_prompt.replace(
                "{rag_context}",
                memory_context
            )
        else:
            enhanced_prompt = enhanced_prompt.replace("{rag_context}", "No relevant memories found.")
    
    if "{history_context}" in enhanced_prompt:
        if conversation_history:
            enhanced_prompt = enhanced_prompt.replace(
                "{history_context}",
                conversation_history
            )
        else:
            enhanced_prompt = enhanced_prompt.replace("{history_context}", "No recent conversation history.")
    
    # Also support OLD placeholders for backward compatibility
    if "{memory_context}" in enhanced_prompt:
        if memory_context:
            enhanced_prompt = enhanced_prompt.replace(
                "{memory_context}", 
                f"\n\n## Relevant Context from Memory:\n{memory_context}\n"
            )
        else:
            enhanced_prompt = enhanced_prompt.replace("{memory_context}", "")
    
    if "{search_results}" in enhanced_prompt:
        if search_results:
            enhanced_prompt = enhanced_prompt.replace(
                "{search_results}",
                f"\n\n## Search Results:\n{search_results}\n"
            )
        else:
            enhanced_prompt = enhanced_prompt.replace("{search_results}", "")
    
    # Build messages list
    messages = [
        Message.system(enhanced_prompt),
        Message.user(user_message)
    ]
    
    return messages


def run_llm_boost(
    user_message: str,
    provider: BaseProvider,
    model_name: Optional[str] = None,
    use_search: bool = True,
    use_memory: bool = True,
    incognito: bool = False,
    temperature: float = 0.7,
    max_reflections: int = 1,
    memory_manager: Optional[MemoryManager] = None
) -> str:
    """
    Run the LLM Boost pipeline.
    
    This is the main entry point that orchestrates:
    1. Context building (memory + search)
    2. Prompt engineering
    3. LLM execution
    4. Reflection loop (self-improvement)
    5. Memory saving
    
    Args:
        user_message: The user's input message
        provider: LLM provider instance (NetworkProvider or MLXProvider)
        model_name: Optional model name override
        use_search: Whether to use web search for context
        use_memory: Whether to use memory for context and saving
        incognito: If True, don't save to memory (privacy mode)
        temperature: Sampling temperature
        max_reflections: Maximum number of reflection iterations
        memory_manager: Optional MemoryManager instance (created if not provided)
        
    Returns:
        The LLM's response text
    """
    # Initialize memory manager if needed
    if use_memory and memory_manager is None:
        memory_manager = MemoryManager()
    
    # Set incognito mode
    if memory_manager:
        memory_manager.incognito = incognito
    
    # ==========================================
    # 1. CONTEXT BUILDING
    # ==========================================
    memory_context = ""
    search_results = ""
    
    # Get memory context
    if use_memory and memory_manager:
        try:
            memory_context = memory_manager.get_context(user_message)
        except Exception as e:
            print(f"Warning: Failed to get memory context: {e}")
    
    # Get search results
    if use_search:
        try:
            search_results = perform_search(user_message)
        except Exception as e:
            print(f"Warning: Search failed: {e}")
            search_results = ""
    
    # ==========================================
    # 2. PROMPT ENGINEERING
    # ==========================================
    system_prompt = load_prompt("system")
    
    # Add placeholders if not present
    if "{memory_context}" not in system_prompt:
        if memory_context:
            system_prompt += "\n\n{memory_context}"
    
    if "{search_results}" not in system_prompt:
        if search_results:
            system_prompt += "\n\n{search_results}"
    
    # Build initial messages
    messages = build_messages(
        user_message=user_message,
        system_prompt=system_prompt,
        memory_context=memory_context,
        search_results=search_results
    )
    
    # ==========================================
    # 3. EXECUTION
    # ==========================================
    response = provider.generate(messages, temperature=temperature)
    
    # ==========================================
    # 3.5 PYTHON CODE EXECUTION
    # ==========================================
    # Check if the model wrote Python code
    python_code = extract_python_code(response)
    if python_code:
        # Execute the code
        code_result = execute_python_code(python_code)
        
        # Append result to context and prompt for final answer
        continuation_messages = build_messages(
            user_message=user_message,
            system_prompt=system_prompt,
            memory_context=memory_context,
            search_results=search_results
        )
        
        # Add the response with code and execution result
        continuation_messages.append(Message.assistant(response))
        continuation_messages.append(Message.user(
            f"[System]: Your Python code has been executed:\n\n{code_result}\n\n"
            "Please use this result to provide your final answer. "
            "Format your response with :final answer: for the user-facing response."
        ))
        
        # Get final response
        response = provider.generate(continuation_messages, temperature=temperature)
    
    # ==========================================
    # 4. REFLECTION LOOP
    # ==========================================
    reflection_count = 0
    
    while reflection_count < max_reflections:
        # Check confidence
        confidence = extract_confidence(response)
        
        if confidence is None or confidence >= 8:
            # Good enough or no confidence score found
            break
        
        # Low confidence - trigger critique
        reflection_count += 1
        
        # Load critique prompt
        critique_prompt = load_prompt("critique")
        
        # Build critique messages
        critique_messages = [
            Message.system(critique_prompt),
            Message.user(f"Original question: {user_message}\n\nMy response:\n{response}\n\nPlease critique and improve.")
        ]
        
        # Get critique
        critique = provider.generate(critique_messages, temperature=temperature)
        
        # Re-run with critique
        improved_messages = build_messages(
            user_message=user_message,
            system_prompt=system_prompt,
            memory_context=memory_context,
            search_results=search_results
        )
        
        # Add the original response and critique as context
        improved_messages.insert(-1, Message.assistant(response))
        improved_messages.insert(-1, Message.user(
            f"I've reviewed my response and received this critique:\n{critique}\n\nPlease provide an improved response."
        ))
        
        response = provider.generate(improved_messages, temperature=temperature)
    
    # ==========================================
    # 5. SAVING
    # ==========================================
    if use_memory and memory_manager:
        try:
            memory_manager.save_interaction(user_message, response)
        except Exception as e:
            print(f"Warning: Failed to save interaction: {e}")
    
    # ==========================================
    # 6. RETURN
    # ==========================================
    return response


def run_llm_boost_simple(
    user_message: str,
    provider: BaseProvider,
    temperature: float = 0.7
) -> str:
    """
    Simplified version of run_llm_boost without search or memory.
    
    Useful for quick queries or when you only want raw LLM output.
    
    Args:
        user_message: The user's input message
        provider: LLM provider instance
        temperature: Sampling temperature
        
    Returns:
        The LLM's response text
    """
    system_prompt = load_prompt("system")
    
    messages = [
        Message.system(system_prompt),
        Message.user(user_message)
    ]
    
    return provider.generate(messages, temperature=temperature)


def run_llm_boost_with_history(
    user_message: str,
    provider: BaseProvider,
    conversation_history: list,
    use_search: bool = True,
    use_memory: bool = True,
    temperature: float = 0.7
) -> Tuple[str, list]:
    """
    Run LLM Boost with existing conversation history.
    
    Args:
        user_message: The user's input message
        provider: LLM provider instance
        conversation_history: List of previous Message objects
        use_search: Whether to use web search
        use_memory: Whether to use memory
        temperature: Sampling temperature
        
    Returns:
        Tuple of (response text, updated conversation history)
    """
    # Build context
    memory_context = ""
    search_results = ""
    
    if use_memory:
        memory_manager = MemoryManager()
        try:
            memory_context = memory_manager.get_context(user_message)
        except Exception:
            pass
    
    if use_search:
        try:
            search_results = perform_search(user_message)
        except Exception:
            pass
    
    # Build messages
    system_prompt = load_prompt("system")
    
    messages = [Message.system(system_prompt)]
    messages.extend(conversation_history)
    messages.append(Message.user(user_message))
    
    # Generate response
    response = provider.generate(messages, temperature=temperature)
    
    # Update history
    updated_history = conversation_history.copy()
    updated_history.append(Message.user(user_message))
    updated_history.append(Message.assistant(response))
    
    # Save to memory
    if use_memory:
        try:
            memory_manager.save_interaction(user_message, response)
        except Exception:
            pass
    
    return response, updated_history