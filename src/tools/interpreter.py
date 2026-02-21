"""
Python Code Interpreter for LLM Boost.
Provides a "Soft Sandbox" for safe code execution.
"""

import sys
import io
import re
import contextlib
from typing import Optional, Tuple
from dataclasses import dataclass


# ============================================
# SECURITY CONFIGURATION
# ============================================

# Blocked modules/keywords - file system and network access
BLOCKED_MODULES = [
    'os', 'subprocess', 'shutil', 'sys', 'open', 'input',
    'socket', 'requests', 'urllib', 'urllib2', 'urllib3',
    'ftplib', 'telnetlib', 'poplib', 'smtplib', 'imaplib',
    'nntplib', 'pickle', 'marshal', 'shelve', 'dbm',
    'sqlite3', 'multiprocessing', 'threading', '_thread',
    'ctypes', 'importlib', 'builtins', '__import__',
    'eval', 'exec', 'compile', 'globals', 'locals',
    'getattr', 'setattr', 'delattr', 'hasattr',
    '__builtins__', '__class__', '__base__', '__subclasses__',
]

# Allowed modules for safe execution
ALLOWED_MODULES = [
    'math', 'numpy', 'np', 'datetime', 'time', 'calendar',
    'random', 'statistics', 'fractions', 'decimal',
    'itertools', 'functools', 'operator', 'collections',
    're', 'string', 'textwrap', 'unicodedata',
    'copy', 'pprint', 'json', 'typing',
]


class SecurityError(Exception):
    """Raised when code contains dangerous patterns."""
    pass


@dataclass
class ExecutionResult:
    """Result of code execution."""
    success: bool
    output: str
    error: Optional[str] = None
    execution_time: Optional[float] = None


class PythonREPL:
    """
    A sandboxed Python interpreter for LLM code execution.
    
    Features:
    - Soft sandbox: Blocks dangerous modules/keywords
    - Output capturing via stdout redirection
    - Timeout support (optional)
    
    Security Note:
    This is a "soft sandbox" suitable for local MVP use.
    For production, use Docker containerization or similar
    hard isolation.
    """
    
    def __init__(self, timeout: int = 30):
        """
        Initialize the Python REPL.
        
        Args:
            timeout: Maximum execution time in seconds (not enforced in MVP)
        """
        self.timeout = timeout
        self._globals = self._build_safe_globals()
    
    def _build_safe_globals(self) -> dict:
        """Build a safe global namespace for execution."""
        import math
        import datetime
        import random
        import statistics
        import itertools
        import functools
        import collections
        import re as re_module
        import json
        import decimal
        import fractions
        import copy
        import pprint
        import string
        import textwrap
        import typing
        
        safe_globals = {
            # Built-in functions (safe subset)
            'abs': abs, 'all': all, 'any': any,
            'bin': bin, 'bool': bool,
            'chr': chr, 'complex': complex,
            'dict': dict, 'divmod': divmod,
            'enumerate': enumerate,
            'filter': filter, 'float': float, 'format': format,
            'frozenset': frozenset,
            'hex': hex,
            'int': int, 'isinstance': isinstance, 'iter': iter,
            'len': len, 'list': list,
            'map': map, 'max': max, 'min': min,
            'next': next,
            'oct': oct, 'ord': ord,
            'pow': pow, 'print': print,
            'range': range, 'repr': repr, 'reversed': reversed, 'round': round,
            'set': set, 'sorted': sorted, 'str': str, 'sum': sum, 'super': super,
            'tuple': tuple, 'type': type,
            'zip': zip,
            'True': True, 'False': False, 'None': None,
            
            # Safe modules
            'math': math,
            'datetime': datetime,
            'random': random,
            'statistics': statistics,
            'itertools': itertools,
            'functools': functools,
            'collections': collections,
            're': re_module,
            'json': json,
            'decimal': decimal,
            'fractions': fractions,
            'copy': copy,
            'pprint': pprint,
            'string': string,
            'textwrap': textwrap,
            'typing': typing,
        }
        
        # Try to add numpy if available
        try:
            import numpy as np
            safe_globals['numpy'] = np
            safe_globals['np'] = np
        except ImportError:
            pass
        
        return safe_globals
    
    def _scan_for_danger(self, code: str) -> None:
        """
        Scan code for dangerous patterns.
        
        Args:
            code: Python code to scan
            
        Raises:
            SecurityError: If dangerous patterns are found
        """
        code_lower = code.lower()
        
        # Check for blocked modules/keywords
        for blocked in BLOCKED_MODULES:
            # Check for import statements
            patterns = [
                rf'\bimport\s+{blocked}\b',
                rf'\bfrom\s+{blocked}\b',
                rf'\b{blocked}\s*\.',
                rf'\b{blocked}\s*\(',
            ]
            for pattern in patterns:
                if re.search(pattern, code_lower):
                    raise SecurityError(
                        f"Dangerous module blocked: '{blocked}'. "
                        "The interpreter is restricted to math and logic operations only."
                    )
        
        # Check for dangerous function calls
        dangerous_calls = [
            r'__import__\s*\(',
            r'eval\s*\(',
            r'exec\s*\(',
            r'compile\s*\(',
            r'open\s*\(',
            r'input\s*\(',
        ]
        
        for pattern in dangerous_calls:
            if re.search(pattern, code_lower):
                raise SecurityError(
                    f"Dangerous function call blocked. "
                    "The interpreter is restricted to math and logic operations only."
                )
        
        # Check for attribute access tricks
        attr_tricks = [
            r'__class__',
            r'__base__',
            r'__subclasses__',
            r'__builtins__',
            r'__globals__',
            r'__code__',
            r'__mro__',
        ]
        
        for trick in attr_tricks:
            if trick in code:
                raise SecurityError(
                    f"Potentially dangerous attribute access blocked: '{trick}'"
                )
    
    def execute(self, code: str) -> ExecutionResult:
        """
        Execute Python code in a sandboxed environment.
        
        Args:
            code: Python code to execute
            
        Returns:
            ExecutionResult with output or error
        """
        import time
        
        # Clean the code
        code = code.strip()
        if not code:
            return ExecutionResult(success=True, output="")
        
        # Security scan
        try:
            self._scan_for_danger(code)
        except SecurityError as e:
            return ExecutionResult(
                success=False,
                output="",
                error=str(e)
            )
        
        # Execute with stdout capture
        start_time = time.time()
        
        try:
            # Create string buffer for stdout
            stdout_buffer = io.StringIO()
            
            # Build local namespace
            local_vars = {}
            
            # Execute with redirected stdout
            with contextlib.redirect_stdout(stdout_buffer):
                exec(code, self._globals, local_vars)
            
            # Get output
            output = stdout_buffer.getvalue()
            
            # If there's a result from the last expression, include it
            if not output.strip() and local_vars:
                # Try to find a meaningful result
                for key, value in reversed(list(local_vars.items())):
                    if not key.startswith('_') and value is not None:
                        output = str(value)
                        break
            
            execution_time = time.time() - start_time
            
            return ExecutionResult(
                success=True,
                output=output,
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return ExecutionResult(
                success=False,
                output="",
                error=f"{type(e).__name__}: {str(e)}",
                execution_time=execution_time
            )
    
    def execute_and_format(self, code: str) -> str:
        """
        Execute code and return a formatted result string.
        
        Args:
            code: Python code to execute
            
        Returns:
            Formatted result string suitable for LLM context
        """
        result = self.execute(code)
        
        if result.success:
            output = result.output.strip() if result.output else "(No output)"
            time_info = f" [Executed in {result.execution_time:.3f}s]" if result.execution_time else ""
            return f"[Python Output]{time_info}:\n{output}"
        else:
            return f"[Python Error]: {result.error}"


# Convenience function for quick execution
def run_python(code: str) -> str:
    """
    Quick function to execute Python code.
    
    Args:
        code: Python code to execute
        
    Returns:
        Output string or error message
    """
    repl = PythonREPL()
    return repl.execute_and_format(code)