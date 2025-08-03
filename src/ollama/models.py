import asyncio
from typing import List, Optional, Dict, Any, Callable, Awaitable
from datetime import datetime

from .client import OllamaClient
from .config import get_ollama_config
from .types import OllamaModel, PullProgress
from .exceptions import OllamaError, ModelNotFoundError
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from shared.utils import format_size, sanitize_model_name


class ModelManager:
    """High-level model management for Ollama."""
    
    def __init__(self):
        self.config = get_ollama_config()
        self._pull_tasks: Dict[str, asyncio.Task] = {}
    
    async def list_models(self, force_refresh: bool = False) -> List[OllamaModel]:
        """Get list of available models."""
        return await self.config.get_available_models(force_refresh=force_refresh)
    
    async def get_model_by_name(self, name: str) -> Optional[OllamaModel]:
        """Get a specific model by name."""
        models = await self.list_models()
        return next((model for model in models if model.name == name), None)
    
    async def model_exists(self, name: str) -> bool:
        """Check if a model exists locally."""
        return await self.config.validate_model(name)
    
    async def get_model_info(self, name: str) -> Optional[Dict[str, Any]]:
        """Get detailed model information."""
        return await self.config.get_model_info(name)
    
    async def get_models_by_size(self, ascending: bool = True) -> List[OllamaModel]:
        """Get models sorted by size."""
        models = await self.list_models()
        return sorted(models, key=lambda m: m.size, reverse=not ascending)
    
    async def get_recent_models(self, limit: int = 5) -> List[OllamaModel]:
        """Get recently modified models."""
        models = await self.list_models()
        try:
            # Sort by modified_at date
            sorted_models = sorted(
                models,
                key=lambda m: datetime.fromisoformat(m.modified_at.replace('Z', '+00:00')),
                reverse=True
            )
            return sorted_models[:limit]
        except Exception:
            # Fallback to original order if date parsing fails
            return models[:limit]
    
    async def search_models(self, query: str) -> List[OllamaModel]:
        """Search models by name."""
        models = await self.list_models()
        query_lower = query.lower()
        
        return [
            model for model in models
            if query_lower in model.name.lower() or 
               query_lower in sanitize_model_name(model.name).lower()
        ]
    
    async def get_memory_usage(self) -> Dict[str, Any]:
        """Get current memory usage information."""
        return await self.config.get_memory_usage()
    
    async def get_total_storage_used(self) -> Dict[str, Any]:
        """Calculate total storage used by all models."""
        models = await self.list_models()
        
        total_bytes = sum(model.size for model in models)
        
        return {
            'total_models': len(models),
            'total_bytes': total_bytes,
            'total_formatted': format_size(total_bytes),
            'average_size_bytes': total_bytes // len(models) if models else 0,
            'largest_model': max(models, key=lambda m: m.size) if models else None,
            'smallest_model': min(models, key=lambda m: m.size) if models else None
        }
    
    async def pull_model(
        self,
        model_name: str,
        progress_callback: Optional[Callable[[PullProgress], Awaitable[None]]] = None
    ) -> bool:
        """Pull/download a model with progress tracking."""
        try:
            # Check if already pulling this model
            if model_name in self._pull_tasks:
                existing_task = self._pull_tasks[model_name]
                if not existing_task.done():
                    await existing_task
                    return True
            
            # Create pull task
            task = asyncio.create_task(
                self._pull_model_task(model_name, progress_callback)
            )
            self._pull_tasks[model_name] = task
            
            result = await task
            
            # Clean up completed task
            if model_name in self._pull_tasks:
                del self._pull_tasks[model_name]
            
            return result
            
        except Exception as e:
            # Clean up failed task
            if model_name in self._pull_tasks:
                del self._pull_tasks[model_name]
            raise OllamaError(f"Failed to pull model {model_name}: {str(e)}")
    
    async def _pull_model_task(
        self,
        model_name: str,
        progress_callback: Optional[Callable[[PullProgress], Awaitable[None]]]
    ) -> bool:
        """Internal task for pulling a model."""
        try:
            await self.config.pull_model(model_name, progress_callback)
            return True
        except Exception as e:
            raise e
    
    def is_pulling(self, model_name: str) -> bool:
        """Check if a model is currently being pulled."""
        task = self._pull_tasks.get(model_name)
        return task is not None and not task.done()
    
    def cancel_pull(self, model_name: str) -> bool:
        """Cancel an ongoing model pull."""
        task = self._pull_tasks.get(model_name)
        if task and not task.done():
            task.cancel()
            del self._pull_tasks[model_name]
            return True
        return False
    
    async def delete_model(self, model_name: str) -> bool:
        """Delete a model."""
        try:
            # Check if model exists
            if not await self.model_exists(model_name):
                raise ModelNotFoundError(f"Model {model_name} not found")
            
            return await self.config.delete_model(model_name)
            
        except Exception as e:
            if isinstance(e, OllamaError):
                raise e
            raise OllamaError(f"Failed to delete model {model_name}: {str(e)}")
    
    async def get_model_suggestions(self, partial_name: str = "") -> List[str]:
        """Get model name suggestions based on partial input."""
        # Common Ollama models that users might want to pull
        common_models = [
            "llama3.2:1b",
            "llama3.2:3b", 
            "llama3.2",
            "llama3.1",
            "llama3.1:7b",
            "llama3.1:13b",
            "llama3.1:70b",
            "codellama",
            "codellama:7b",
            "codellama:13b",
            "mistral",
            "mistral:7b",
            "gemma",
            "gemma:2b",
            "gemma:7b",
            "phi3",
            "phi3:mini",
            "qwen2",
            "qwen2:0.5b",
            "qwen2:1.5b",
            "qwen2:7b",
            "all-minilm",
            "nomic-embed-text"
        ]
        
        # Get currently installed models
        installed_models = await self.list_models()
        installed_names = [model.name for model in installed_models]
        
        # Filter suggestions
        if partial_name:
            partial_lower = partial_name.lower()
            suggestions = [
                name for name in common_models
                if partial_lower in name.lower() and name not in installed_names
            ]
        else:
            suggestions = [
                name for name in common_models
                if name not in installed_names
            ]
        
        return suggestions[:10]  # Limit to 10 suggestions
    
    async def validate_model_for_chat(self, model_name: str) -> bool:
        """Validate if a model is suitable for chat operations."""
        try:
            model_info = await self.get_model_info(model_name)
            if not model_info:
                return False
            
            # Check if model has chat template
            template = model_info.get('template', '')
            details = model_info.get('details', {})
            
            # Most chat models have templates or are instruction-tuned
            if template or (details and 'instruct' in str(details).lower()):
                return True
            
            # Allow common chat model patterns
            model_lower = model_name.lower()
            chat_patterns = ['chat', 'instruct', 'llama', 'mistral', 'gemma', 'phi', 'qwen']
            
            return any(pattern in model_lower for pattern in chat_patterns)
            
        except Exception:
            # If we can't verify, assume it's usable
            return True
    
    async def get_default_model(self) -> Optional[str]:
        """Get the configured default model."""
        default = self.config.default_model
        
        # Validate that the default model still exists
        if default and await self.model_exists(default):
            return default
        
        # If no default or invalid, try to pick a good one
        models = await self.list_models()
        if not models:
            return None
        
        # Prefer common chat models
        chat_models = [
            m for m in models
            if any(pattern in m.name.lower() 
                  for pattern in ['llama', 'mistral', 'gemma', 'phi', 'qwen'])
        ]
        
        if chat_models:
            # Sort by size (prefer smaller for responsiveness)
            chat_models.sort(key=lambda m: m.size)
            return chat_models[0].name
        
        # Fallback to any available model
        return models[0].name
    
    async def set_default_model(self, model_name: Optional[str]) -> bool:
        """Set the default model."""
        if model_name and not await self.model_exists(model_name):
            raise ModelNotFoundError(f"Model {model_name} not found")
        
        return self.config.update_default_model(model_name)
    
    async def cleanup(self):
        """Cancel all ongoing operations and cleanup."""
        # Cancel all pull tasks
        for task in self._pull_tasks.values():
            if not task.done():
                task.cancel()
        
        # Wait for cancellations to complete
        if self._pull_tasks:
            await asyncio.gather(*self._pull_tasks.values(), return_exceptions=True)
        
        self._pull_tasks.clear()


# Global model manager instance
_model_manager_instance: Optional[ModelManager] = None


def get_model_manager() -> ModelManager:
    """Get the global model manager instance."""
    global _model_manager_instance
    if _model_manager_instance is None:
        _model_manager_instance = ModelManager()
    return _model_manager_instance


async def cleanup_model_manager():
    """Cleanup the global model manager instance."""
    global _model_manager_instance
    if _model_manager_instance:
        await _model_manager_instance.cleanup()
        _model_manager_instance = None