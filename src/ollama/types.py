from typing import Optional, Dict, Any, List, Union
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from shared.types import ChatMessage as BaseChatMessage


class ModelStatus(Enum):
    """Model status in memory."""
    LOADED = "loaded"
    LOADING = "loading"
    UNLOADED = "unloaded"
    ERROR = "error"


@dataclass
class OllamaMessage:
    """Ollama API message format."""
    role: str
    content: str
    images: Optional[List[str]] = None


@dataclass
class GenerateRequest:
    """Request for Ollama generate endpoint."""
    model: str
    prompt: str
    system: Optional[str] = None
    template: Optional[str] = None
    context: Optional[List[int]] = None
    stream: bool = False
    raw: bool = False
    format: Optional[str] = None
    options: Optional[Dict[str, Any]] = None
    keep_alive: Optional[Union[str, int]] = None


@dataclass
class ChatRequest:
    """Request for Ollama chat endpoint."""
    model: str
    messages: List[OllamaMessage]
    tools: Optional[List[Dict[str, Any]]] = None
    stream: bool = False
    format: Optional[str] = None
    options: Optional[Dict[str, Any]] = None
    keep_alive: Optional[Union[str, int]] = None


@dataclass
class GenerateResponse:
    """Response from Ollama generate endpoint."""
    model: str
    created_at: str
    response: str
    done: bool
    context: Optional[List[int]] = None
    total_duration: Optional[int] = None
    load_duration: Optional[int] = None
    prompt_eval_count: Optional[int] = None
    prompt_eval_duration: Optional[int] = None
    eval_count: Optional[int] = None
    eval_duration: Optional[int] = None


@dataclass
class ChatResponse:
    """Response from Ollama chat endpoint."""
    model: str
    created_at: str
    message: OllamaMessage
    done: bool
    total_duration: Optional[int] = None
    load_duration: Optional[int] = None
    prompt_eval_count: Optional[int] = None
    prompt_eval_duration: Optional[int] = None
    eval_count: Optional[int] = None
    eval_duration: Optional[int] = None


@dataclass
class ModelDetails:
    """Detailed model information."""
    parent_model: str
    format: str
    family: str
    families: Optional[List[str]] = None
    parameter_size: Optional[str] = None
    quantization_level: Optional[str] = None


@dataclass
class OllamaModel:
    """Ollama model information."""
    name: str
    size: int
    digest: str
    modified_at: str
    details: Optional[ModelDetails] = None
    
    @property
    def size_gb(self) -> float:
        """Get size in gigabytes."""
        return self.size / (1024 ** 3)
    
    @property
    def size_mb(self) -> float:
        """Get size in megabytes."""
        return self.size / (1024 ** 2)
    
    @property
    def display_name(self) -> str:
        """Get display-friendly model name."""
        return self.name.replace(':', ' ')


@dataclass
class ModelListResponse:
    """Response from model list endpoint."""
    models: List[OllamaModel]


@dataclass
class ModelShowResponse:
    """Response from model show endpoint."""
    modelfile: str
    parameters: str
    template: str
    details: ModelDetails
    model_info: Optional[Dict[str, Any]] = None


@dataclass
class ModelInMemory:
    """Model currently loaded in memory."""
    name: str
    model: str
    size: int
    digest: str
    details: ModelDetails
    expires_at: str
    size_vram: int


@dataclass
class ProcessListResponse:
    """Response from process list endpoint."""
    models: List[ModelInMemory]


@dataclass
class PullProgress:
    """Progress information for model pulling."""
    status: str
    digest: Optional[str] = None
    total: Optional[int] = None
    completed: Optional[int] = None
    
    @property
    def progress_percent(self) -> float:
        """Calculate progress percentage."""
        if self.total and self.completed:
            return (self.completed / self.total) * 100
        return 0.0


@dataclass
class EmbedRequest:
    """Request for embedding generation."""
    model: str
    input: Union[str, List[str]]
    truncate: Optional[bool] = None
    options: Optional[Dict[str, Any]] = None
    keep_alive: Optional[Union[str, int]] = None


@dataclass
class EmbedResponse:
    """Response from embed endpoint."""
    model: str
    embeddings: List[List[float]]
    total_duration: Optional[int] = None
    load_duration: Optional[int] = None
    prompt_eval_count: Optional[int] = None


@dataclass
class BlobInfo:
    """Information about a blob."""
    digest: str
    size: int