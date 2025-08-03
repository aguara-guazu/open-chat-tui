import asyncio
import uuid
from typing import Optional

from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Input, Static, Header, Footer
from textual.reactive import reactive
from textual.binding import Binding

# Import widgets and managers (keeping your existing imports)
from .widgets.options_dialog import OptionsDialog
from .widgets.chat_message import EnhancedChatMessage, StreamingChatMessage, SystemMessage, ErrorMessage
from .widgets.status_bar import StatusBar
from .commands.handlers import get_command_handler

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from ollama import (
    get_chat_manager, get_model_manager, get_ollama_config,
    OllamaError, ConnectionError as OllamaConnectionError
)
from shared.types import ChatRole


class MenuOption(Static):
    """A widget for menu options in the sidebar."""

    def __init__(self, label: str, action: Optional[str] = None) -> None:
        super().__init__(label)
        self.add_class("menu-option")
        self.action = action


class ChatApp(App):
    """Enhanced chat TUI application with full Ollama integration."""

    CSS_PATH = "chat.tcss"  # Updated to use .tcss extension
    TITLE = "Chat TUI - Ollama Interface"

    # Responsive breakpoints for different terminal sizes
    HORIZONTAL_BREAKPOINTS = [
        (0, "-very-narrow"),    # 0-59 columns: hide sidebar
        (60, "-narrow"),        # 60-79 columns: narrow sidebar
        (80, "-normal"),        # 80+ columns: normal layout
    ]

    # Proper Textual bindings
    BINDINGS = [
        Binding("ctrl+q", "quit", "Quit", show=False, priority=True),
        Binding("ctrl+o", "show_options", "Options", tooltip="Open options dialog"),
        Binding("ctrl+r", "refresh_models", "Refresh", tooltip="Refresh models"),
        Binding("ctrl+c", "cancel_operation", "Cancel", show=False),
        Binding("f1", "show_help", "Help", tooltip="Show help"),
        Binding("escape", "handle_escape", "", show=False),
        Binding("ctrl+n", "new_chat", "New Chat", tooltip="Start new chat"),
        Binding("ctrl+shift+c", "clear_chat", "Clear", tooltip="Clear chat history"),
    ]

    current_model: reactive[Optional[str]] = reactive(None)
    is_streaming: reactive[bool] = reactive(False)

    def __init__(self):
        super().__init__()

        # Initialize managers
        self.chat_manager = get_chat_manager()
        self.model_manager = get_model_manager()
        self.ollama_config = get_ollama_config()
        self.command_handler = get_command_handler()

        # Session management
        self.current_session_id = str(uuid.uuid4())
        self.current_session = None

        # Streaming state
        self._current_streaming_message: Optional[StreamingChatMessage] = None

    def compose(self) -> ComposeResult:
        """Compose the app layout."""
        yield Header()

        with Horizontal():
            # Left sidebar menu
            with Vertical(id="sidebar"):
                yield Static("🤖 Llama Term", classes="sidebar-title")
                yield MenuOption("💬 New Chat", "new_chat")
                yield MenuOption("🔄 Refresh Models", "refresh_models")
                yield MenuOption("⚙️ Settings", "settings")
                yield MenuOption("❓ Help", "help")

            # Main chat area
            with Vertical(id="main-area"):
                # Chat messages area
                with VerticalScroll(id="chat-area"):
                    yield SystemMessage("Welcome to Llama Term!")
                    yield SystemMessage("A powerful TUI for chatting with Ollama models.")
                    yield SystemMessage("Press Ctrl+O to open settings and select a model.")
                    yield SystemMessage("Type your messages below or use /help for commands.")

                # Input area at bottom
                yield Input(
                    placeholder="Type your message here... (Ctrl+O: Options, Ctrl+Q: Quit)",
                    id="message-input"
                )

        # Status bar and footer
        yield StatusBar()
        yield Footer()

    async def on_mount(self) -> None:
        """Initialize the app when mounted."""
        # Initialize chat session
        try:
            default_model = await self.model_manager.get_default_model()
            if default_model:
                self.current_session = await self.chat_manager.create_session(
                    self.current_session_id,
                    model_name=default_model
                )
                self.current_model = default_model
                await self.add_system_message(f"Using model: {default_model}")
            else:
                await self.add_system_message("No model selected. Press Ctrl+O to choose a model.")

        except Exception as e:
            await self.add_error_message(f"Initialization error: {str(e)}")

        # Test connection
        asyncio.create_task(self._test_initial_connection())

    # Textual Action Methods - this is how Textual wants us to handle keys!
    async def action_show_options(self) -> None:
        """Show the options dialog."""
        try:
            dialog = OptionsDialog(on_model_changed=self._on_model_changed)
            result = await self.push_screen(dialog)

            if result == "saved":
                await self.add_system_message("Settings saved successfully")

                # Update status bar
                status_bar = self.query_one(StatusBar)
                await status_bar.force_update()

        except Exception as e:
            await self.add_error_message(f"Error opening options: {str(e)}")

    async def action_refresh_models(self) -> None:
        """Refresh the model list."""
        try:
            await self.model_manager.list_models(force_refresh=True)
            await self.add_system_message("Model list refreshed")

            # Update status bar
            status_bar = self.query_one(StatusBar)
            await status_bar.force_update()

        except Exception as e:
            await self.add_error_message(f"Error refreshing models: {str(e)}")

    async def action_show_help(self) -> None:
        """Show help information."""
        help_text = self.command_handler.get_help_text()
        await self.add_system_message(f"Help:\n{help_text}")

        # Also show key bindings
        bindings_help = "Key Bindings:\n"
        for binding in self.BINDINGS:
            if binding.show and binding.description:
                key_display = binding.key.replace("ctrl+", "Ctrl+").replace("shift+", "Shift+")
                bindings_help += f"  {key_display}: {binding.description}\n"

        await self.add_system_message(bindings_help)

    async def action_new_chat(self) -> None:
        """Start a new chat session."""
        await self.clear_chat()

    async def action_clear_chat(self) -> None:
        """Clear the chat history."""
        try:
            # Clear UI
            chat_area = self.query_one("#chat-area")
            chat_area.remove_children()

            # Clear session
            if self.current_session:
                self.chat_manager.clear_session(self.current_session_id)

            # Add welcome messages
            chat_area.mount(SystemMessage("Chat cleared"))
            chat_area.mount(SystemMessage("Start a new conversation!"))

        except Exception as e:
            await self.add_error_message(f"Error clearing chat: {str(e)}")

    def action_cancel_operation(self) -> None:
        """Cancel current operation."""
        if self.is_streaming and self._current_streaming_message:
            # Cancel streaming
            self.is_streaming = False
            self._current_streaming_message = None
            asyncio.create_task(self.add_system_message("Operation cancelled"))

    def action_handle_escape(self) -> None:
        """Handle escape key press."""
        # Try to close any modal screens first
        if hasattr(self, 'screen_stack') and len(self.screen_stack) > 1:
            self.pop_screen()
        else:
            # Cancel any ongoing operations
            self.action_cancel_operation()

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle message submission."""
        message = event.value.strip()

        # Don't process empty messages
        if not message:
            return

        # Clear input immediately
        event.input.clear()

        try:
            # Handle commands
            if message.startswith('/'):
                handled = await self.command_handler.handle_command(message, self)
                if handled:
                    return

            # Ensure we have a session and model
            if not self.current_session:
                await self.add_error_message("No active session. Please select a model first.")
                return

            # Add user message to UI
            await self.add_user_message(message)

            # Send message to Ollama with streaming
            await self._send_message_to_ollama(message)

        except Exception as e:
            await self.add_error_message(f"Error processing message: {str(e)}")

    async def _test_initial_connection(self) -> None:
        """Test initial connection to Ollama."""
        try:
            is_connected = await self.ollama_config.test_connection()
            if is_connected:
                await self.add_system_message("✅ Connected to Ollama server")
            else:
                await self.add_system_message("❌ Failed to connect to Ollama server")
        except Exception as e:
            await self.add_error_message(f"Connection test failed: {str(e)}")

    async def _send_message_to_ollama(self, message: str) -> None:
        """Send message to Ollama and handle streaming response."""
        if not self.current_session:
            await self.add_error_message("No active session")
            return

        try:
            self.is_streaming = True

            # Create streaming message widget
            self._current_streaming_message = StreamingChatMessage(
                role=ChatRole.ASSISTANT,
                sender_name="Assistant"
            )

            # Add to chat area
            chat_area = self.query_one("#chat-area")
            chat_area.mount(self._current_streaming_message)
            chat_area.scroll_end()

            # Define streaming callback
            async def stream_callback(chunk: str):
                if self._current_streaming_message:
                    self._current_streaming_message.stream_chunk(chunk)
                    chat_area.scroll_end()

            # Send message with streaming
            full_response = await self.chat_manager.send_message(
                self.current_session_id,
                message,
                stream=True,
                stream_callback=stream_callback
            )

            # Finish streaming
            if self._current_streaming_message:
                self._current_streaming_message.finish_streaming({
                    'model': self.current_model,
                    'full_response_length': len(full_response)
                })

        except OllamaConnectionError:
            await self.add_error_message("Connection to Ollama failed. Check your settings.")
        except OllamaError as e:
            await self.add_error_message(f"Ollama error: {str(e)}")
        except Exception as e:
            await self.add_error_message(f"Unexpected error: {str(e)}")
        finally:
            self.is_streaming = False
            self._current_streaming_message = None

    async def add_user_message(self, content: str) -> None:
        """Add a user message to the chat area."""
        chat_area = self.query_one("#chat-area")
        message = EnhancedChatMessage(
            content=content,
            role=ChatRole.USER,
            sender_name="You"
        )
        chat_area.mount(message)
        chat_area.scroll_end()

    async def add_system_message(self, content: str) -> None:
        """Add a system message to the chat area."""
        chat_area = self.query_one("#chat-area")
        message = SystemMessage(content)
        chat_area.mount(message)
        chat_area.scroll_end()

    async def add_error_message(self, content: str) -> None:
        """Add an error message to the chat area."""
        chat_area = self.query_one("#chat-area")
        message = ErrorMessage(content)
        chat_area.mount(message)
        chat_area.scroll_end()

    async def _on_model_changed(self, model_name: str) -> None:
        """Handle model change from options dialog."""
        try:
            # Update current session model
            if self.current_session:
                await self.chat_manager.change_session_model(
                    self.current_session_id,
                    model_name
                )
            else:
                # Create new session with the model
                self.current_session = await self.chat_manager.create_session(
                    self.current_session_id,
                    model_name=model_name
                )

            self.current_model = model_name
            await self.add_system_message(f"Model changed to: {model_name}")

            # Update status bar
            status_bar = self.query_one(StatusBar)
            await status_bar.show_model_changed(model_name)

        except Exception as e:
            await self.add_error_message(f"Error changing model: {str(e)}")

    async def clear_chat(self) -> None:
        """Clear the chat history."""
        await self.action_clear_chat()

    async def refresh_models(self) -> None:
        """Refresh the model list."""
        await self.action_refresh_models()

    async def show_help(self, help_text: Optional[str] = None) -> None:
        """Show help information."""
        if help_text:
            await self.add_system_message(f"Help:\n{help_text}")
        else:
            await self.action_show_help()

    async def set_current_model(self, model_name: str) -> None:
        """Set the current model programmatically."""
        await self._on_model_changed(model_name)

    async def cleanup(self) -> None:
        """Cleanup resources before exiting."""
        try:
            # Cleanup chat manager
            if hasattr(self.chat_manager, 'cleanup'):
                await self.chat_manager.cleanup()

            # Cleanup model manager
            if hasattr(self.model_manager, 'cleanup'):
                await self.model_manager.cleanup()

            # Cleanup config
            if hasattr(self.ollama_config, 'cleanup'):
                await self.ollama_config.cleanup()

        except Exception as e:
            # Don't fail on cleanup errors
            pass

    # Menu option handlers - using proper event handling
    async def on_static_clicked(self, event) -> None:
        """Handle menu option clicks."""
        if isinstance(event.static, MenuOption):
            action = event.static.action

            if action == "new_chat":
                await self.action_new_chat()
            elif action == "refresh_models":
                await self.action_refresh_models()
            elif action == "settings":
                await self.action_show_options()
            elif action == "help":
                await self.action_show_help()


def run_chat_app():
    """Entry point to run the chat application."""
    app = ChatApp()

    try:
        app.run()
    except KeyboardInterrupt:
        pass
    finally:
        # Ensure cleanup
        try:
            asyncio.run(app.cleanup())
        except:
            pass


if __name__ == "__main__":
    run_chat_app()