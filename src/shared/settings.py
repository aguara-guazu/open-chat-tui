"""
Settings management for the Ollama Chat TUI application.
This module handles application settings, configuration, and persistence.
"""

import json
import os
from typing import Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass, asdict

from .types import ConnectionConfig, AppSettings


@dataclass
class Settings:
    """Application settings with default values."""

    # Connection settings
    base_url: str = "http://localhost:11434"
    timeout: int = 30
    max_retries: int = 3

    # Chat settings
    default_model: Optional[str] = None
    system_prompt: Optional[str] = None
    max_history_size: int = 1000
    auto_save_chat: bool = True

    # UI settings
    theme: str = "dark"
    show_timestamps: bool = True
    show_model_info: bool = True

    # Performance settings
    stream_responses: bool = True
    keep_alive: str = "5m"

    def to_dict(self) -> Dict[str, Any]:
        """Convert settings to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Settings':
        """Create settings from dictionary."""
        # Filter out any keys that don't exist in the dataclass
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_data = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered_data)

    def get_connection_config(self) -> ConnectionConfig:
        """Get connection configuration."""
        return ConnectionConfig(
            base_url=self.base_url,
            timeout=self.timeout,
            max_retries=self.max_retries
        )

    def get_app_settings(self) -> AppSettings:
        """Get application settings."""
        return AppSettings(
            connection=self.get_connection_config(),
            default_model=self.default_model,
            theme=self.theme,
            auto_save_chat=self.auto_save_chat,
            max_history_size=self.max_history_size
        )


class SettingsManager:
    """Manages application settings persistence and access."""

    def __init__(self, config_dir: Optional[str] = None):
        """Initialize settings manager."""
        if config_dir is None:
            # Use user's config directory
            home = Path.home()
            self.config_dir = home / ".config" / "ollama-chat-tui"
        else:
            self.config_dir = Path(config_dir)

        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.config_file = self.config_dir / "settings.json"

        # Load existing settings or create defaults
        self._settings = self._load_settings()

    def _load_settings(self) -> Settings:
        """Load settings from file."""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    data = json.load(f)
                return Settings.from_dict(data)
            except (json.JSONDecodeError, TypeError, ValueError) as e:
                print(f"Warning: Could not load settings: {e}")
                print("Using default settings.")

        return Settings()

    def _save_settings(self) -> bool:
        """Save settings to file."""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self._settings.to_dict(), f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving settings: {e}")
            return False

    def get_setting(self, key: str, default: Any = None) -> Any:
        """Get a specific setting value."""
        return getattr(self._settings, key, default)

    def set_setting(self, key: str, value: Any) -> bool:
        """Set a specific setting value."""
        if hasattr(self._settings, key):
            setattr(self._settings, key, value)
            return self._save_settings()
        else:
            print(f"Warning: Unknown setting key: {key}")
            return False

    def get_all_settings(self) -> Settings:
        """Get all settings."""
        return self._settings

    def update_settings(self, **kwargs) -> bool:
        """Update multiple settings at once."""
        updated = False
        for key, value in kwargs.items():
            if hasattr(self._settings, key):
                setattr(self._settings, key, value)
                updated = True
            else:
                print(f"Warning: Unknown setting key: {key}")

        if updated:
            return self._save_settings()
        return False

    def reset_to_defaults(self) -> bool:
        """Reset all settings to defaults."""
        self._settings = Settings()
        return self._save_settings()

    def get_connection_config(self) -> ConnectionConfig:
        """Get connection configuration."""
        return self._settings.get_connection_config()

    def get_app_settings(self) -> AppSettings:
        """Get application settings."""
        return self._settings.get_app_settings()

    def export_settings(self, file_path: str) -> bool:
        """Export settings to a file."""
        try:
            with open(file_path, 'w') as f:
                json.dump(self._settings.to_dict(), f, indent=2)
            return True
        except Exception as e:
            print(f"Error exporting settings: {e}")
            return False

    def import_settings(self, file_path: str) -> bool:
        """Import settings from a file."""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)

            self._settings = Settings.from_dict(data)
            return self._save_settings()
        except Exception as e:
            print(f"Error importing settings: {e}")
            return False


# Global settings manager instance
_settings_manager_instance: Optional[SettingsManager] = None


def get_settings_manager() -> SettingsManager:
    """Get the global settings manager instance."""
    global _settings_manager_instance
    if _settings_manager_instance is None:
        _settings_manager_instance = SettingsManager()
    return _settings_manager_instance


def get_settings() -> Settings:
    """Get current application settings."""
    return get_settings_manager().get_all_settings()


def update_settings(**kwargs) -> bool:
    """Update application settings."""
    return get_settings_manager().update_settings(**kwargs)


def reset_settings() -> bool:
    """Reset settings to defaults."""
    return get_settings_manager().reset_to_defaults()


# Cleanup function for testing
def cleanup_settings_manager():
    """Cleanup the global settings manager instance."""
    global _settings_manager_instance
    _settings_manager_instance = None