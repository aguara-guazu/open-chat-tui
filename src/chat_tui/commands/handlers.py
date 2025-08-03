"""Command handlers for chat application commands - Updated for Textual compatibility."""

import asyncio
from typing import Dict, List, Optional, Any, Callable
from textual.app import App

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from ollama import get_chat_manager, get_model_manager, get_ollama_config
from shared.types import ChatRole


class CommandHandler:
    """Handles slash commands for the chat application."""

    def __init__(self):
        self.commands: Dict[str, Callable] = {}
        self.descriptions: Dict[str, str] = {}
        self._register_default_commands()

    def register_command(self, command: str, handler: Callable, description: str = ""):
        """Register a command handler."""
        self.commands[command.lower()] = handler
        if description:
            self.descriptions[command.lower()] = description

    def unregister_command(self, command: str):
        """Unregister a command handler."""
        command = command.lower()
        self.commands.pop(command, None)
        self.descriptions.pop(command, None)

    async def handle_command(self, command_text: str, app: App) -> bool:
        """Handle a command. Returns True if handled, False otherwise."""
        if not command_text.startswith('/'):
            return False

        # Parse command and arguments
        parts = command_text[1:].split()
        if not parts:
            return False

        command = parts[0].lower()
        args = parts[1:] if len(parts) > 1 else []

        if command in self.commands:
            try:
                handler = self.commands[command]

                # Check if handler expects app parameter
                import inspect
                sig = inspect.signature(handler)
                params = list(sig.parameters.keys())

                if 'app' in params:
                    if inspect.iscoroutinefunction(handler):
                        await handler(args, app)
                    else:
                        handler(args, app)
                else:
                    if inspect.iscoroutinefunction(handler):
                        await handler(args)
                    else:
                        handler(args)

                return True

            except Exception as e:
                # Use proper Textual notification system
                if hasattr(app, 'add_system_message'):
                    await app.add_system_message(f"Command error: {str(e)}")
                elif hasattr(app, 'notify'):
                    app.notify(f"Command error: {str(e)}")
                return True  # Still handled, even if there was an error

        return False

    def get_help_text(self) -> str:
        """Get help text for all commands."""
        if not self.descriptions:
            return "No commands available"

        help_lines = [f"/{cmd}: {desc}" for cmd, desc in self.descriptions.items()]
        return "\n".join(help_lines)

    def list_commands(self) -> List[str]:
        """Get list of available commands."""
        return list(self.commands.keys())

    def _register_default_commands(self):
        """Register default chat commands."""

        # Exit command - now uses Textual's action system
        self.register_command(
            "quit",
            self._quit_command,
            "Exit the application (same as Ctrl+Q)"
        )

        # Help command
        self.register_command(
            "help",
            self._help_command,
            "Show available commands"
        )

        # Model commands
        self.register_command(
            "models",
            self._list_models_command,
            "List available models"
        )

        self.register_command(
            "model",
            self._set_model_command,
            "Set current model (usage: /model <model_name>)"
        )

        # Chat commands
        self.register_command(
            "clear",
            self._clear_chat_command,
            "Clear chat history (same as Ctrl+Shift+C)"
        )

        self.register_command(
            "new",
            self._new_chat_command,
            "Start new chat (same as Ctrl+N)"
        )

        self.register_command(
            "history",
            self._show_history_command,
            "Show chat history summary"
        )

        # Connection commands
        self.register_command(
            "connect",
            self._connect_command,
            "Test connection to Ollama"
        )

        self.register_command(
            "status",
            self._status_command,
            "Show system status"
        )

        # Configuration commands
        self.register_command(
            "config",
            self._config_command,
            "Show configuration"
        )

        self.register_command(
            "url",
            self._set_url_command,
            "Set Ollama URL (usage: /url <url>)"
        )

        # Options command - now uses Textual's action system
        self.register_command(
            "options",
            self._options_command,
            "Open options dialog (same as Ctrl+O)"
        )

    def _quit_command(self, args: List[str], app: App):
        """Handle quit command - use Textual's built-in quit action."""
        app.action_quit()

    async def _options_command(self, args: List[str], app: App):
        """Handle options command - use Textual's action system."""
        if hasattr(app, 'action_show_options'):
            await app.action_show_options()
        else:
            await app.add_system_message("Options dialog not available")

    async def _new_chat_command(self, args: List[str], app: App):
        """Handle new chat command - use Textual's action system."""
        if hasattr(app, 'action_new_chat'):
            await app.action_new_chat()
        else:
            await app.add_system_message("New chat action not available")

    async def _clear_chat_command(self, args: List[str], app: App):
        """Handle clear chat command - use Textual's action system."""
        if hasattr(app, 'action_clear_chat'):
            await app.action_clear_chat()
        else:
            await app.add_system_message("Clear chat action not available")

    async def _help_command(self, args: List[str], app: App):
        """Handle help command."""
        help_text = self.get_help_text()

        if hasattr(app, 'add_system_message'):
            await app.add_system_message(f"Available commands:\n{help_text}")
        elif hasattr(app, 'notify'):
            app.notify(f"Commands: {', '.join(self.list_commands())}")

    async def _list_models_command(self, args: List[str], app: App):
        """Handle models list command."""
        try:
            model_manager = get_model_manager()
            models = await model_manager.list_models()

            if not models:
                message = "No models available. Use /model pull <name> to download models."
            else:
                model_list = "\n".join([
                    f"• {model.name} ({model.size_gb:.1f}GB)"
                    for model in models
                ])
                message = f"Available models:\n{model_list}"

            if hasattr(app, 'add_system_message'):
                await app.add_system_message(message)
            elif hasattr(app, 'notify'):
                app.notify(f"Found {len(models)} models")

        except Exception as e:
            error_msg = f"Error listing models: {str(e)}"
            if hasattr(app, 'add_system_message'):
                await app.add_system_message(error_msg)
            elif hasattr(app, 'notify'):
                app.notify(error_msg)

    async def _set_model_command(self, args: List[str], app: App):
        """Handle model set command."""
        if not args:
            if hasattr(app, 'add_system_message'):
                await app.add_system_message("Usage: /model <model_name>")
            return

        model_name = args[0]

        try:
            model_manager = get_model_manager()

            # Check if model exists
            if not await model_manager.model_exists(model_name):
                message = f"Model '{model_name}' not found. Use /models to see available models."
                if hasattr(app, 'add_system_message'):
                    await app.add_system_message(message)
                return

            # Set as default model
            await model_manager.set_default_model(model_name)

            # Update app if it has a method for this
            if hasattr(app, 'set_current_model'):
                await app.set_current_model(model_name)

            message = f"Model changed to: {model_name}"
            if hasattr(app, 'add_system_message'):
                await app.add_system_message(message)
            elif hasattr(app, 'notify'):
                app.notify(message)

        except Exception as e:
            error_msg = f"Error setting model: {str(e)}"
            if hasattr(app, 'add_system_message'):
                await app.add_system_message(error_msg)
            elif hasattr(app, 'notify'):
                app.notify(error_msg)
    
    async def _show_history_command(self, args: List[str], app: App):
        """Handle history command."""
        try:
            chat_manager = get_chat_manager()
            sessions = chat_manager.list_sessions()
            
            if not sessions:
                message = "No chat history available"
            else:
                # Show summary of current session
                if hasattr(app, 'current_session_id'):
                    session = chat_manager.get_session(app.current_session_id)
                    if session:
                        message_count = session.get_message_count()
                        model = session.model_name
                        message = f"Current session: {message_count} messages using {model}"
                    else:
                        message = "No active session"
                else:
                    message = f"Found {len(sessions)} chat sessions"
            
            if hasattr(app, 'add_system_message'):
                await app.add_system_message(message)
            elif hasattr(app, 'notify'):
                app.notify(message)
                
        except Exception as e:
            error_msg = f"Error getting history: {str(e)}"
            if hasattr(app, 'add_system_message'):
                await app.add_system_message(error_msg)
    
    async def _connect_command(self, args: List[str], app: App):
        """Handle connect command."""
        try:
            config = get_ollama_config()
            is_connected = await config.test_connection()
            
            if is_connected:
                message = f"✅ Connected to Ollama at {config.base_url}"
            else:
                message = f"❌ Failed to connect to Ollama at {config.base_url}"
            
            if hasattr(app, 'add_system_message'):
                await app.add_system_message(message)
            elif hasattr(app, 'notify'):
                app.notify(message)
                
        except Exception as e:
            error_msg = f"Connection test error: {str(e)}"
            if hasattr(app, 'add_system_message'):
                await app.add_system_message(error_msg)
    
    async def _status_command(self, args: List[str], app: App):
        """Handle status command."""
        try:
            config = get_ollama_config()
            model_manager = get_model_manager()
            
            # Get status info
            is_connected = await config.test_connection()
            models = await model_manager.list_models()
            default_model = await model_manager.get_default_model()
            memory_info = await config.get_memory_usage()
            
            status_lines = [
                f"Connection: {'✅ Connected' if is_connected else '❌ Disconnected'}",
                f"Server: {config.base_url}",
                f"Models: {len(models)} available",
                f"Default model: {default_model or 'None'}",
                f"Memory usage: {memory_info.get('total_memory_gb', 0):.1f}GB"
            ]
            
            message = "\n".join(status_lines)
            
            if hasattr(app, 'add_system_message'):
                await app.add_system_message(message)
            elif hasattr(app, 'notify'):
                app.notify("Status info displayed")
                
        except Exception as e:
            error_msg = f"Error getting status: {str(e)}"
            if hasattr(app, 'add_system_message'):
                await app.add_system_message(error_msg)
    
    async def _config_command(self, args: List[str], app: App):
        """Handle config command."""
        try:
            config = get_ollama_config()
            settings = config.settings.settings
            
            config_lines = [
                f"Base URL: {config.base_url}",
                f"Timeout: {config.connection_config.timeout}s",
                f"Max retries: {config.connection_config.max_retries}",
                f"Default model: {settings.default_model or 'None'}",
                f"Theme: {settings.theme}",
                f"Auto-save: {settings.auto_save_chat}"
            ]
            
            message = "Current configuration:\n" + "\n".join(config_lines)
            
            if hasattr(app, 'add_system_message'):
                await app.add_system_message(message)
            elif hasattr(app, 'notify'):
                app.notify("Configuration displayed")
                
        except Exception as e:
            error_msg = f"Error getting config: {str(e)}"
            if hasattr(app, 'add_system_message'):
                await app.add_system_message(error_msg)
    
    async def _set_url_command(self, args: List[str], app: App):
        """Handle URL set command."""
        if not args:
            if hasattr(app, 'add_system_message'):
                await app.add_system_message("Usage: /url <url>")
            return
        
        new_url = args[0]
        
        try:
            config = get_ollama_config()
            success = config.update_base_url(new_url)
            
            if success:
                message = f"URL updated to: {new_url}"
                # Test new connection
                is_connected = await config.test_connection()
                if is_connected:
                    message += " ✅ Connection successful"
                else:
                    message += " ❌ Connection failed"
            else:
                message = f"Failed to update URL to: {new_url}"
            
            if hasattr(app, 'add_system_message'):
                await app.add_system_message(message)
            elif hasattr(app, 'notify'):
                app.notify(message)
                
        except Exception as e:
            error_msg = f"Error setting URL: {str(e)}"
            if hasattr(app, 'add_system_message'):
                await app.add_system_message(error_msg)


# Global command handler instance
_command_handler_instance: Optional[CommandHandler] = None


def get_command_handler() -> CommandHandler:
    """Get the global command handler instance."""
    global _command_handler_instance
    if _command_handler_instance is None:
        _command_handler_instance = CommandHandler()
    return _command_handler_instance