import asyncio
from typing import Optional, List, Dict, Any
from dataclasses import dataclass

from .client import OllamaClient
from .exceptions import ConnectionError, OllamaError
from .types import OllamaModel, ModelListResponse
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from shared.settings import get_settings, Settings
from shared.types import ConnectionConfig, ConnectionStatus


@dataclass
class OllamaConfig:
    """Ollama-specific configuration manager."""
    
    def __init__(self):
        self._settings = get_settings()
        self._client: Optional[OllamaClient] = None
        self._cached_models: Optional[List[OllamaModel]] = None
        self._connection_status = ConnectionStatus.DISCONNECTED
    
    @property
    def settings(self) -> Settings:
        """Get the underlying settings manager."""
        return self._settings
    
    @property
    def connection_config(self) -> ConnectionConfig:
        """Get current connection configuration."""
        return self._settings.settings.connection
    
    @property
    def base_url(self) -> str:
        """Get Ollama base URL."""
        return self.connection_config.base_url
    
    @property
    def default_model(self) -> Optional[str]:
        """Get default model name."""
        return self._settings.settings.default_model
    
    @property
    def connection_status(self) -> ConnectionStatus:
        """Get current connection status."""
        return self._connection_status
    
    def get_client(self) -> OllamaClient:
        """Get or create Ollama client."""
        if self._client is None:
            self._client = OllamaClient(self.connection_config)
        return self._client
    
    async def test_connection(self) -> bool:
        """Test connection to Ollama server."""
        try:
            self._connection_status = ConnectionStatus.CONNECTING
            client = self.get_client()
            
            is_healthy = await client.health_check()
            
            if is_healthy:
                self._connection_status = ConnectionStatus.CONNECTED
                return True
            else:
                self._connection_status = ConnectionStatus.ERROR
                return False
                
        except Exception as e:
            self._connection_status = ConnectionStatus.ERROR
            return False
    
    async def get_available_models(self, force_refresh: bool = False) -> List[OllamaModel]:
        """Get list of available models."""
        if self._cached_models is None or force_refresh:
            try:
                client = self.get_client()
                response = await client.list_models()
                self._cached_models = response.models
            except Exception as e:
                # Return empty list if we can't fetch models
                self._cached_models = []
        
        return self._cached_models or []
    
    async def validate_model(self, model_name: str) -> bool:
        """Check if a model is available."""
        try:
            models = await self.get_available_models()
            return any(model.name == model_name for model in models)
        except Exception:
            return False
    
    def update_base_url(self, base_url: str) -> bool:
        """Update Ollama base URL."""
        try:
            # Validate URL format
            if not base_url.startswith(('http://', 'https://')):
                base_url = f"http://{base_url}"
            
            # Update settings
            success = self._settings.update_connection(base_url=base_url)
            
            if success:
                # Reset client to use new URL
                if self._client:
                    asyncio.create_task(self._client.close())
                    self._client = None
                
                # Reset cached data
                self._cached_models = None
                self._connection_status = ConnectionStatus.DISCONNECTED
            
            return success
            
        except Exception as e:
            return False
    
    def update_default_model(self, model_name: Optional[str]) -> bool:
        """Update default model."""
        try:
            return self._settings.update(default_model=model_name)
        except Exception:
            return False
    
    def update_connection_settings(self, **kwargs) -> bool:
        """Update connection settings."""
        try:
            success = self._settings.update_connection(**kwargs)
            
            if success:
                # Reset client if connection settings changed
                if self._client:
                    asyncio.create_task(self._client.close())
                    self._client = None
                
                # Reset status
                self._connection_status = ConnectionStatus.DISCONNECTED
            
            return success
            
        except Exception:
            return False
    
    async def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a model."""
        try:
            client = self.get_client()
            response = await client.show_model(model_name)
            
            return {
                'name': model_name,
                'modelfile': response.modelfile,
                'parameters': response.parameters,
                'template': response.template,
                'details': response.details.__dict__ if response.details else None
            }
            
        except Exception as e:
            return None
    
    async def pull_model(self, model_name: str, progress_callback=None):
        """Pull/download a model with optional progress callback."""
        try:
            client = self.get_client()
            
            async for progress in client.pull_model(model_name):
                if progress_callback:
                    await progress_callback(progress)
                
                # If download is complete, refresh model cache
                if progress.status == "success":
                    self._cached_models = None
                    
        except Exception as e:
            raise OllamaError(f"Failed to pull model {model_name}: {str(e)}")
    
    async def delete_model(self, model_name: str) -> bool:
        """Delete a model."""
        try:
            client = self.get_client()
            success = await client.delete_model(model_name)
            
            if success:
                # Refresh model cache
                self._cached_models = None
                
                # Update default model if it was deleted
                if self.default_model == model_name:
                    self.update_default_model(None)
            
            return success
            
        except Exception as e:
            return False
    
    async def get_memory_usage(self) -> Dict[str, Any]:
        """Get memory usage information."""
        try:
            client = self.get_client()
            response = await client.list_running_models()
            
            total_memory = sum(model.size_vram for model in response.models)
            
            return {
                'loaded_models': len(response.models),
                'total_memory_bytes': total_memory,
                'total_memory_gb': total_memory / (1024 ** 3),
                'models': [
                    {
                        'name': model.name,
                        'memory_bytes': model.size_vram,
                        'memory_gb': model.size_vram / (1024 ** 3),
                        'expires_at': model.expires_at
                    }
                    for model in response.models
                ]
            }
            
        except Exception as e:
            return {
                'loaded_models': 0,
                'total_memory_bytes': 0,
                'total_memory_gb': 0.0,
                'models': []
            }
    
    async def cleanup(self):
        """Cleanup resources."""
        if self._client:
            await self._client.close()
            self._client = None
        
        self._cached_models = None
        self._connection_status = ConnectionStatus.DISCONNECTED
    
    def reset_to_defaults(self) -> bool:
        """Reset configuration to defaults."""
        try:
            success = self._settings.reset_to_defaults()
            
            if success:
                # Cleanup current client
                if self._client:
                    asyncio.create_task(self._client.close())
                    self._client = None
                
                # Reset cached data
                self._cached_models = None
                self._connection_status = ConnectionStatus.DISCONNECTED
            
            return success
            
        except Exception:
            return False


# Global configuration instance
_config_instance: Optional[OllamaConfig] = None


def get_ollama_config() -> OllamaConfig:
    """Get the global Ollama configuration instance."""
    global _config_instance
    if _config_instance is None:
        _config_instance = OllamaConfig()
    return _config_instance


async def cleanup_ollama_config():
    """Cleanup the global configuration instance."""
    global _config_instance
    if _config_instance:
        await _config_instance.cleanup()
        _config_instance = None