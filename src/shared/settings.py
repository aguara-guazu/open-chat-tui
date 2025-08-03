import json
import os
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import asdict

from .types import AppSettings, ConnectionConfig


class Settings:
    """Manages application settings persistence."""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config_dir = Path.home() / ".config" / "llama-term"
        self.config_file = config_file or self.config_dir / "settings.json"
        self._settings: Optional[AppSettings] = None
        
        # Ensure config directory exists
        self.config_dir.mkdir(parents=True, exist_ok=True)
    
    @property
    def settings(self) -> AppSettings:
        """Get current settings, loading from file if needed."""
        if self._settings is None:
            self._settings = self.load()
        return self._settings
    
    def load(self) -> AppSettings:
        """Load settings from file or create defaults."""
        try:
            if self.config_file.exists():
                with open(self.config_file, 'r') as f:
                    data = json.load(f)
                    
                # Convert connection dict to ConnectionConfig if needed
                if 'connection' in data and isinstance(data['connection'], dict):
                    data['connection'] = ConnectionConfig(**data['connection'])
                
                return AppSettings(**data)
            else:
                # Create default settings
                return self._create_default_settings()
                
        except Exception as e:
            print(f"Error loading settings: {e}")
            return self._create_default_settings()
    
    def save(self, settings: Optional[AppSettings] = None) -> bool:
        """Save settings to file."""
        try:
            settings_to_save = settings or self.settings
            
            # Convert to dict for JSON serialization
            data = asdict(settings_to_save)
            
            with open(self.config_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            
            self._settings = settings_to_save
            return True
            
        except Exception as e:
            print(f"Error saving settings: {e}")
            return False
    
    def update(self, **kwargs) -> bool:
        """Update specific settings values."""
        try:
            current = self.settings
            
            # Update the current settings
            for key, value in kwargs.items():
                if hasattr(current, key):
                    setattr(current, key, value)
            
            return self.save(current)
            
        except Exception as e:
            print(f"Error updating settings: {e}")
            return False
    
    def update_connection(self, **kwargs) -> bool:
        """Update connection-specific settings."""
        try:
            current = self.settings
            
            # Update connection settings
            for key, value in kwargs.items():
                if hasattr(current.connection, key):
                    setattr(current.connection, key, value)
            
            return self.save(current)
            
        except Exception as e:
            print(f"Error updating connection settings: {e}")
            return False
    
    def reset_to_defaults(self) -> bool:
        """Reset settings to defaults."""
        try:
            self._settings = self._create_default_settings()
            return self.save()
        except Exception as e:
            print(f"Error resetting settings: {e}")
            return False
    
    def _create_default_settings(self) -> AppSettings:
        """Create default application settings."""
        return AppSettings(
            connection=ConnectionConfig(),
            default_model=None,
            theme="dark",
            auto_save_chat=True,
            max_history_size=1000
        )
    
    @property
    def config_path(self) -> str:
        """Get the configuration file path."""
        return str(self.config_file)


# Global settings instance
_settings_instance: Optional[Settings] = None


def get_settings() -> Settings:
    """Get the global settings instance."""
    global _settings_instance
    if _settings_instance is None:
        _settings_instance = Settings()
    return _settings_instance


def reload_settings() -> Settings:
    """Reload settings from file."""
    global _settings_instance
    _settings_instance = None
    return get_settings()