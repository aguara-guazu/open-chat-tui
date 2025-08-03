import json
import asyncio
from typing import Optional, Dict, Any, List, AsyncIterator, Union
from contextlib import asynccontextmanager

import httpx

from .types import (
    GenerateRequest, GenerateResponse, ChatRequest, ChatResponse,
    ModelListResponse, ModelShowResponse, ProcessListResponse,
    EmbedRequest, EmbedResponse, OllamaModel, PullProgress
)
from .exceptions import (
    OllamaError, ConnectionError, TimeoutError, StreamingError,
    handle_response_error
)
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from shared.types import ConnectionConfig


class OllamaClient:
    """Async HTTP client for Ollama API."""
    
    def __init__(self, config: ConnectionConfig):
        self.config = config
        self._client: Optional[httpx.AsyncClient] = None
    
    @asynccontextmanager
    async def _get_client(self):
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.config.base_url,
                timeout=httpx.Timeout(self.config.timeout),
                follow_redirects=True
            )
        
        try:
            yield self._client
        except Exception as e:
            # Don't close client on error, reuse it
            raise e
    
    async def close(self):
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None
    
    async def health_check(self) -> bool:
        """Check if Ollama server is available."""
        try:
            async with self._get_client() as client:
                response = await client.get("/")
                return response.status_code == 200
        except Exception:
            return False
    
    async def generate(self, request: GenerateRequest) -> GenerateResponse:
        """Generate text completion."""
        try:
            async with self._get_client() as client:
                response = await client.post(
                    "/api/generate",
                    json=self._request_to_dict(request),
                    headers={"Content-Type": "application/json"}
                )
                
                if response.status_code != 200:
                    error_data = {}
                    try:
                        error_data = response.json()
                    except:
                        pass
                    raise handle_response_error(response.status_code, error_data)
                
                data = response.json()
                return GenerateResponse(**data)
                
        except httpx.TimeoutException:
            raise TimeoutError("Request timed out")
        except httpx.ConnectError:
            raise ConnectionError("Unable to connect to Ollama server")
        except Exception as e:
            if isinstance(e, OllamaError):
                raise e
            raise OllamaError(f"Generate request failed: {str(e)}")
    
    async def generate_stream(self, request: GenerateRequest) -> AsyncIterator[GenerateResponse]:
        """Generate text completion with streaming."""
        request.stream = True
        
        try:
            async with self._get_client() as client:
                async with client.stream(
                    "POST",
                    "/api/generate",
                    json=self._request_to_dict(request),
                    headers={"Content-Type": "application/json"}
                ) as response:
                    
                    if response.status_code != 200:
                        error_data = {}
                        try:
                            error_data = await response.json()
                        except:
                            pass
                        raise handle_response_error(response.status_code, error_data)
                    
                    async for line in response.aiter_lines():
                        if line.strip():
                            try:
                                data = json.loads(line)
                                yield GenerateResponse(**data)
                            except json.JSONDecodeError:
                                continue
                            
        except httpx.TimeoutException:
            raise TimeoutError("Stream timed out")
        except httpx.ConnectError:
            raise ConnectionError("Unable to connect to Ollama server")
        except Exception as e:
            if isinstance(e, OllamaError):
                raise e
            raise StreamingError(f"Generate stream failed: {str(e)}")
    
    async def chat(self, request: ChatRequest) -> ChatResponse:
        """Send chat messages."""
        try:
            async with self._get_client() as client:
                response = await client.post(
                    "/api/chat",
                    json=self._request_to_dict(request),
                    headers={"Content-Type": "application/json"}
                )
                
                if response.status_code != 200:
                    error_data = {}
                    try:
                        error_data = response.json()
                    except:
                        pass
                    raise handle_response_error(response.status_code, error_data)
                
                data = response.json()
                return ChatResponse(**data)
                
        except httpx.TimeoutException:
            raise TimeoutError("Chat request timed out")
        except httpx.ConnectError:
            raise ConnectionError("Unable to connect to Ollama server")
        except Exception as e:
            if isinstance(e, OllamaError):
                raise e
            raise OllamaError(f"Chat request failed: {str(e)}")
    
    async def chat_stream(self, request: ChatRequest) -> AsyncIterator[ChatResponse]:
        """Send chat messages with streaming."""
        request.stream = True
        
        try:
            async with self._get_client() as client:
                async with client.stream(
                    "POST",
                    "/api/chat", 
                    json=self._request_to_dict(request),
                    headers={"Content-Type": "application/json"}
                ) as response:
                    
                    if response.status_code != 200:
                        error_data = {}
                        try:
                            error_data = await response.json()
                        except:
                            pass
                        raise handle_response_error(response.status_code, error_data)
                    
                    async for line in response.aiter_lines():
                        if line.strip():
                            try:
                                data = json.loads(line)
                                yield ChatResponse(**data)
                            except json.JSONDecodeError:
                                continue
                                
        except httpx.TimeoutException:
            raise TimeoutError("Chat stream timed out")
        except httpx.ConnectError:
            raise ConnectionError("Unable to connect to Ollama server")
        except Exception as e:
            if isinstance(e, OllamaError):
                raise e
            raise StreamingError(f"Chat stream failed: {str(e)}")
    
    async def list_models(self) -> ModelListResponse:
        """List available models."""
        try:
            async with self._get_client() as client:
                response = await client.get("/api/tags")
                
                if response.status_code != 200:
                    error_data = {}
                    try:
                        error_data = response.json()
                    except:
                        pass
                    raise handle_response_error(response.status_code, error_data)
                
                data = response.json()
                models = [OllamaModel(**model) for model in data.get('models', [])]
                return ModelListResponse(models=models)
                
        except httpx.TimeoutException:
            raise TimeoutError("List models request timed out")
        except httpx.ConnectError:
            raise ConnectionError("Unable to connect to Ollama server")
        except Exception as e:
            if isinstance(e, OllamaError):
                raise e
            raise OllamaError(f"List models failed: {str(e)}")
    
    async def show_model(self, model_name: str, verbose: bool = False) -> ModelShowResponse:
        """Show model information."""
        try:
            async with self._get_client() as client:
                response = await client.post(
                    "/api/show",
                    json={"model": model_name, "verbose": verbose},
                    headers={"Content-Type": "application/json"}
                )
                
                if response.status_code != 200:
                    error_data = {}
                    try:
                        error_data = response.json()
                    except:
                        pass
                    raise handle_response_error(response.status_code, error_data)
                
                data = response.json()
                return ModelShowResponse(**data)
                
        except httpx.TimeoutException:
            raise TimeoutError("Show model request timed out")
        except httpx.ConnectError:
            raise ConnectionError("Unable to connect to Ollama server")
        except Exception as e:
            if isinstance(e, OllamaError):
                raise e
            raise OllamaError(f"Show model failed: {str(e)}")
    
    async def pull_model(self, model_name: str) -> AsyncIterator[PullProgress]:
        """Pull/download a model with progress."""
        try:
            async with self._get_client() as client:
                async with client.stream(
                    "POST",
                    "/api/pull",
                    json={"model": model_name, "stream": True},
                    headers={"Content-Type": "application/json"}
                ) as response:
                    
                    if response.status_code != 200:
                        error_data = {}
                        try:
                            error_data = await response.json()
                        except:
                            pass
                        raise handle_response_error(response.status_code, error_data)
                    
                    async for line in response.aiter_lines():
                        if line.strip():
                            try:
                                data = json.loads(line)
                                yield PullProgress(**data)
                            except json.JSONDecodeError:
                                continue
                                
        except httpx.TimeoutException:
            raise TimeoutError("Pull model timed out")
        except httpx.ConnectError:
            raise ConnectionError("Unable to connect to Ollama server")
        except Exception as e:
            if isinstance(e, OllamaError):
                raise e
            raise StreamingError(f"Pull model failed: {str(e)}")
    
    async def delete_model(self, model_name: str) -> bool:
        """Delete a model."""
        try:
            async with self._get_client() as client:
                response = await client.delete(
                    "/api/delete",
                    json={"model": model_name},
                    headers={"Content-Type": "application/json"}
                )
                
                if response.status_code != 200:
                    error_data = {}
                    try:
                        error_data = response.json()
                    except:
                        pass
                    raise handle_response_error(response.status_code, error_data)
                
                return True
                
        except httpx.TimeoutException:
            raise TimeoutError("Delete model request timed out")
        except httpx.ConnectError:
            raise ConnectionError("Unable to connect to Ollama server")
        except Exception as e:
            if isinstance(e, OllamaError):
                raise e
            raise OllamaError(f"Delete model failed: {str(e)}")
    
    async def list_running_models(self) -> ProcessListResponse:
        """List models currently in memory."""
        try:
            async with self._get_client() as client:
                response = await client.get("/api/ps")
                
                if response.status_code != 200:
                    error_data = {}
                    try:
                        error_data = response.json()
                    except:
                        pass
                    raise handle_response_error(response.status_code, error_data)
                
                data = response.json()
                return ProcessListResponse(**data)
                
        except httpx.TimeoutException:
            raise TimeoutError("List running models timed out")
        except httpx.ConnectError:
            raise ConnectionError("Unable to connect to Ollama server")
        except Exception as e:
            if isinstance(e, OllamaError):
                raise e
            raise OllamaError(f"List running models failed: {str(e)}")
    
    async def embed(self, request: EmbedRequest) -> EmbedResponse:
        """Generate embeddings."""
        try:
            async with self._get_client() as client:
                response = await client.post(
                    "/api/embed",
                    json=self._request_to_dict(request),
                    headers={"Content-Type": "application/json"}
                )
                
                if response.status_code != 200:
                    error_data = {}
                    try:
                        error_data = response.json()
                    except:
                        pass
                    raise handle_response_error(response.status_code, error_data)
                
                data = response.json()
                return EmbedResponse(**data)
                
        except httpx.TimeoutException:
            raise TimeoutError("Embed request timed out")
        except httpx.ConnectError:
            raise ConnectionError("Unable to connect to Ollama server")
        except Exception as e:
            if isinstance(e, OllamaError):
                raise e
            raise OllamaError(f"Embed request failed: {str(e)}")
    
    def _request_to_dict(self, request) -> Dict[str, Any]:
        """Convert request object to dictionary."""
        if hasattr(request, '__dict__'):
            # Filter out None values
            return {k: v for k, v in request.__dict__.items() if v is not None}
        return request
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()