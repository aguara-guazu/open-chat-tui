import asyncio
from typing import Optional, List, Dict, Any

from .client import OllamaClient
from .exceptions import ConnectionError, OllamaError
from .types import OllamaModel, ModelListResponse
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from shared.settings import get_settings_manager, Settings  # FIXED: Import the right functions
from shared.types import ConnectionConfig, ConnectionStatus


class OllamaConfig:  # FIXED: Removed @dataclass decorator
    """Ollama-specific configuration manager."""

    def __init__(self):
        self._settings_manager = get_settings_manager()  # FIXED: Use settings manager
        self._client: Optional[OllamaClient] = None
        self._cached_models: Optional[List[OllamaModel]] = None
        self._connection_status = ConnectionStatus.DISCONNECTED

    @property
    def settings(self) -> Settings:
        """Get the underlying settings."""
        return self._settings_manager.get_all_settings()  # FIXED: Get settings from manager

    @property
    def connection_config(self) -> ConnectionConfig:
        """Get current connection configuration."""
        return self.settings.get_connection_config()  # FIXED: Use settings method

    @property
    def base_url(self) -> str:
        """Get Ollama base URL."""
        return self.connection_config.base_url

    @property
    def default_model(self) -> Optional[str]:
        """Get default model name."""
        return self.settings.default_model

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
            success = self._settings_manager.set_setting('base_url', base_url)

            if success:
                # Cleanup current client to force recreation with new URL
                if self._client:
                    asyncio.create_task(self._client.close())
                    self._client = None

                # Reset connection status
                self._connection_status = ConnectionStatus.DISCONNECTED

            return success

        except Exception:
            return False

    def update_model(self, model_name: str) -> bool:
        """Update default model."""
        try:
            return self._settings_manager.set_setting('default_model', model_name)
        except Exception:
            return False

    async def get_memory_usage(self) -> Dict[str, Any]:
        """Get memory usage information."""
        try:
            # This would typically call an Ollama API endpoint
            # For now, return placeholder data
            return {
                "total_memory_gb": 0.0,
                "used_memory_gb": 0.0,
                "available_memory_gb": 0.0
            }
        except Exception:
            return {
                "total_memory_gb": 0.0,
                "used_memory_gb": 0.0,
                "available_memory_gb": 0.0
            }

    async def save(self) -> bool:
        """Save current configuration."""
        # Settings are automatically saved by the settings manager
        return True

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
            success = self._settings_manager.reset_to_defaults()

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