from enum import Enum
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from datetime import datetime


class ChatRole(Enum):
    """Chat message roles."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class ConnectionStatus(Enum):
    """Connection status indicators."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"


@dataclass
class ChatMessage:
    """Represents a chat message."""
    role: ChatRole
    content: str
    timestamp: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class ModelInfo:
    """Represents information about an Ollama model."""
    name: str
    size: int
    digest: str
    modified_at: datetime
    details: Optional[Dict[str, Any]] = None
    
    @property
    def size_mb(self) -> float:
        """Get size in megabytes."""
        return self.size / (1024 * 1024)
    
    @property
    def size_gb(self) -> float:
        """Get size in gigabytes."""
        return self.size / (1024 * 1024 * 1024)


@dataclass
class ConnectionConfig:
    """Configuration for Ollama connection."""
    base_url: str = "http://localhost:11434"
    timeout: int = 30
    max_retries: int = 3
    
    def __post_init__(self):
        # Ensure URL doesn't end with slash
        self.base_url = self.base_url.rstrip('/')


@dataclass
class AppSettings:
    """Application-wide settings."""
    connection: ConnectionConfig
    default_model: Optional[str] = None
    theme: str = "dark"
    auto_save_chat: bool = True
    max_history_size: int = 1000
    
    def __post_init__(self):
        if isinstance(self.connection, dict):
            self.connection = ConnectionConfig(**self.connection)