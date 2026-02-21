"""
Memory Storage for LLM Boost.
Manages SQLite for structured data and ChromaDB for vector embeddings.
"""

import os
import sqlite3
import json
from datetime import datetime
from typing import Optional, List, Dict, Any

# ChromaDB and embeddings will be imported when needed
try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False


class MemoryStorage:
    """
    Manages both SQLite (structured data) and ChromaDB (vector embeddings).
    Provides a unified interface for storing and retrieving memories.
    """
    
    def __init__(
        self,
        sqlite_path: str = "./data/memory.db",
        chroma_path: str = "./data/chroma",
        embedding_model: str = "all-MiniLM-L6-v2"
    ):
        """
        Initialize memory storage.
        
        Args:
            sqlite_path: Path to SQLite database file
            chroma_path: Path to ChromaDB persistence directory
            embedding_model: Name of the sentence-transformer model to use
        """
        self.sqlite_path = sqlite_path
        self.chroma_path = chroma_path
        self.embedding_model_name = embedding_model
        
        # Ensure directories exist
        sqlite_dir = os.path.dirname(sqlite_path)
        if sqlite_dir:
            os.makedirs(sqlite_dir, exist_ok=True)
        os.makedirs(chroma_path, exist_ok=True)
        
        # Initialize SQLite
        self._init_sqlite()
        
        # Initialize ChromaDB (lazy loading)
        self._chroma_client = None
        self._collection = None
        self._embedding_model = None
    
    def _init_sqlite(self):
        """Initialize SQLite database with required tables."""
        self.conn = sqlite3.connect(self.sqlite_path)
        self.conn.row_factory = sqlite3.Row
        cursor = self.conn.cursor()
        
        # Create memories table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create conversations table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                session_id TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        self.conn.commit()
    
    def _init_chroma(self):
        """Initialize ChromaDB client and collection (lazy loading)."""
        if not CHROMADB_AVAILABLE:
            raise ImportError("ChromaDB is not installed. Run: pip install chromadb")
        
        if self._chroma_client is None:
            self._chroma_client = chromadb.PersistentClient(
                path=self.chroma_path,
                settings=Settings(anonymized_telemetry=False)
            )
            self._collection = self._chroma_client.get_or_create_collection(
                name="llm_boost_memories",
                metadata={"hnsw:space": "cosine"}
            )
    
    def _init_embedding_model(self):
        """Initialize the embedding model (lazy loading)."""
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "sentence-transformers is not installed. "
                "Run: pip install sentence-transformers"
            )
        
        if self._embedding_model is None:
            self._embedding_model = SentenceTransformer(self.embedding_model_name)
    
    def add_memory(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Add a memory to both SQLite and ChromaDB.
        
        Args:
            content: The memory content to store
            metadata: Optional metadata dictionary
            
        Returns:
            The ID of the stored memory
        """
        # Store in SQLite
        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT INTO memories (content, metadata) VALUES (?, ?)",
            (content, json.dumps(metadata or {}))
        )
        self.conn.commit()
        memory_id = cursor.lastrowid
        
        # Store in ChromaDB
        if CHROMADB_AVAILABLE:
            self._init_chroma()
            self._init_embedding_model()
            
            embedding = self._embedding_model.encode(content).tolist()
            self._collection.add(
                ids=[str(memory_id)],
                embeddings=[embedding],
                documents=[content],
                metadatas=[metadata or {}]
            )
        
        return memory_id
    
    def search_similar(
        self,
        query: str,
        n_results: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search for similar memories using vector similarity.
        
        Args:
            query: The query string
            n_results: Number of results to return
            
        Returns:
            List of similar memories with their content and metadata
        """
        if not CHROMADB_AVAILABLE:
            raise ImportError("ChromaDB is not installed. Run: pip install chromadb")
        
        self._init_chroma()
        self._init_embedding_model()
        
        query_embedding = self._embedding_model.encode(query).tolist()
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        
        memories = []
        for i, doc in enumerate(results['documents'][0]):
            memories.append({
                'id': int(results['ids'][0][i]),
                'content': doc,
                'metadata': results['metadatas'][0][i],
                'distance': results['distances'][0][i] if 'distances' in results else None
            })
        
        return memories
    
    def add_conversation(
        self,
        role: str,
        content: str,
        session_id: Optional[str] = None
    ) -> int:
        """
        Add a conversation turn to the database.
        
        Args:
            role: The role (user/assistant/system)
            content: The message content
            session_id: Optional session identifier
            
        Returns:
            The ID of the stored conversation
        """
        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT INTO conversations (role, content, session_id) VALUES (?, ?, ?)",
            (role, content, session_id)
        )
        self.conn.commit()
        return cursor.lastrowid
    
    def get_conversation_history(
        self,
        session_id: Optional[str] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Get conversation history.
        
        Args:
            session_id: Optional session ID to filter by
            limit: Maximum number of messages to return
            
        Returns:
            List of conversation messages
        """
        cursor = self.conn.cursor()
        
        if session_id:
            cursor.execute(
                """
                SELECT role, content, created_at 
                FROM conversations 
                WHERE session_id = ? 
                ORDER BY created_at DESC 
                LIMIT ?
                """,
                (session_id, limit)
            )
        else:
            cursor.execute(
                """
                SELECT role, content, created_at 
                FROM conversations 
                ORDER BY created_at DESC 
                LIMIT ?
                """,
                (limit,)
            )
        
        rows = cursor.fetchall()
        return [dict(row) for row in reversed(rows)]
    
    def clear_memories(self):
        """Clear all memories from both SQLite and ChromaDB."""
        # Clear SQLite
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM memories")
        cursor.execute("DELETE FROM conversations")
        self.conn.commit()
        
        # Clear ChromaDB
        if CHROMADB_AVAILABLE and self._collection is not None:
            self._chroma_client.delete_collection("llm_boost_memories")
            self._collection = None
    
    def close(self):
        """Close all connections."""
        if self.conn:
            self.conn.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class MemoryManager:
    """
    High-level memory management with privacy controls.
    
    Provides a unified interface for storing and retrieving context,
    with built-in support for incognito mode (privacy controls).
    """
    
    def __init__(
        self,
        db_path: str = "chroma_db",
        sqlite_path: str = "./data/memory.db",
        chroma_path: str = "./data/chroma",
        embedding_model: str = "all-MiniLM-L6-v2"
    ):
        """
        Initialize the memory manager.
        
        Args:
            db_path: Legacy parameter for ChromaDB path (deprecated, use chroma_path)
            sqlite_path: Path to SQLite database file
            chroma_path: Path to ChromaDB persistence directory
            embedding_model: Name of the sentence-transformer model
        """
        # Use chroma_path or fall back to db_path
        self.chroma_path = chroma_path if chroma_path != "chroma_db" else db_path
        
        # Initialize the underlying storage
        self.storage = MemoryStorage(
            sqlite_path=sqlite_path,
            chroma_path=self.chroma_path,
            embedding_model=embedding_model
        )
        
        # Track incognito state
        self._incognito = False
    
    @property
    def incognito(self) -> bool:
        """Check if incognito mode is enabled."""
        return self._incognito
    
    @incognito.setter
    def incognito(self, value: bool):
        """Set incognito mode."""
        self._incognito = value
    
    def get_context(self, query: str, n_results: int = 5) -> str:
        """
        Get relevant context from memory for a query.
        
        Searches both ChromaDB (vector similarity) and SQLite
        (conversation history) to provide comprehensive context.
        
        Args:
            query: The query to search for context
            n_results: Maximum number of results to return
            
        Returns:
            Formatted string with relevant context, or empty string if none found.
        """
        if not query or not query.strip():
            return ""
        
        context_parts = []
        
        # Search ChromaDB for similar memories
        try:
            if CHROMADB_AVAILABLE:
                similar_memories = self.storage.search_similar(query, n_results)
                if similar_memories:
                    context_parts.append("### Relevant Memories:")
                    for i, mem in enumerate(similar_memories, 1):
                        context_parts.append(f"{i}. {mem['content']}")
        except Exception:
            pass  # Silently skip if ChromaDB fails
        
        # Get recent conversation history
        try:
            history = self.storage.get_conversation_history(limit=3)
            if history:
                context_parts.append("\n### Recent Conversation:")
                for msg in history:
                    role = msg.get('role', 'unknown').capitalize()
                    content = msg.get('content', '')
                    context_parts.append(f"{role}: {content[:200]}...")
        except Exception:
            pass  # Silently skip if SQLite fails
        
        return "\n".join(context_parts) if context_parts else ""
    
    def save_interaction(
        self,
        user_input: str,
        agent_response: str,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[int]:
        """
        Save an interaction to memory.
        
        CRITICAL: This method includes robust validation and privacy controls.
        - Only saves if both inputs are valid (non-empty)
        - Respects incognito mode - does NOT save when incognito=True
        - The UI layer controls incognito mode, this function enforces it
        
        Args:
            user_input: The user's input message
            agent_response: The agent's response
            session_id: Optional session identifier
            metadata: Optional metadata dictionary
            
        Returns:
            Memory ID if saved, None if skipped (incognito or invalid inputs)
        """
        # CRITICAL: Check incognito mode first
        if self._incognito:
            return None
        
        # Validate inputs - must be non-empty strings
        if not user_input or not isinstance(user_input, str):
            return None
        if not agent_response or not isinstance(agent_response, str):
            return None
        
        # Trim and check for empty strings
        user_input = user_input.strip()
        agent_response = agent_response.strip()
        
        if not user_input or not agent_response:
            return None
        
        # Save to SQLite conversation history
        self.storage.add_conversation("user", user_input, session_id)
        self.storage.add_conversation("assistant", agent_response, session_id)
        
        # Save combined interaction to ChromaDB for semantic search
        combined = f"User: {user_input}\nAssistant: {agent_response}"
        memory_id = self.storage.add_memory(combined, metadata)
        
        return memory_id
    
    def clear_all(self):
        """Clear all stored memories."""
        self.storage.clear_memories()
    
    def close(self):
        """Close all connections."""
        self.storage.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
