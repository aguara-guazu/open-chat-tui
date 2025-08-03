import asyncio
from typing import Optional, Callable, Awaitable

from textual.app import ComposeResult
from textual.containers import Vertical, Horizontal, Grid
from textual.widgets import Static, Input, Button, Label, Switch, TabbedContent, TabPane
from textual.screen import ModalScreen
from textual.reactive import reactive

from .model_selector import ModelSelector
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from ollama import get_ollama_config
from shared.types import ConnectionStatus
from shared.utils import validate_url


class OptionsDialog(ModalScreen):
    """Modal dialog for application options and configuration."""

    DEFAULT_CSS = """
    OptionsDialog {
        align: center middle;
    }

    .dialog-container {
        width: 80;
        height: 30;
        background: $surface;
        border: solid $primary;
        padding: 1;
    }

    .dialog-title {
        text-align: center;
        text-style: bold;
        background: $primary;
        color: $text;
        padding: 0 1;
        margin: 0 0 1 0;
    }

    .connection-status {
        height: auto;
        margin: 1 0;
        padding: 0 1;
        border-left: solid;
    }

    .connection-status.connected {
        border-left: solid $success;
        background: $success-darken-3;
    }

    .connection-status.disconnected {
        border-left: solid $error;
        background: $error-darken-3;
    }

    .connection-status.connecting {
        border-left: solid $warning;
        background: $warning-darken-3;
    }

    .dialog-buttons {
        height: auto;
        layout: horizontal;
        margin: 1 0 0 0;
        align: center bottom;
    }

    .config-section {
        height: auto;
        border: solid $accent;
        padding: 1;
        margin: 1 0;
    }

    TabPane {
        padding: 1;
    }
    """

    connection_status: reactive[ConnectionStatus] = reactive(ConnectionStatus.DISCONNECTED)

    def __init__(self, on_model_changed: Optional[Callable[[str], Awaitable[None]]] = None):
        super().__init__()
        self.ollama_config = get_ollama_config()
        self.on_model_changed = on_model_changed
        self._testing_connection = False

    def compose(self) -> ComposeResult:
        """Compose the options dialog."""
        with Vertical(classes="dialog-container"):
            yield Static("⚙️ Options", classes="dialog-title")

            with TabbedContent():
                # Connection Tab - Direct widget yields
                with TabPane("Connection", id="connection_tab"):
                    with Vertical():
                        yield Static("", id="connection_status", classes="connection-status disconnected")

                        with Vertical(classes="config-section"):
                            yield Label("Ollama Server URL:")
                            yield Input(
                                value=self.ollama_config.base_url,
                                placeholder="http://localhost:11434",
                                id="base_url_input"
                            )

                            yield Label("Connection Timeout (seconds):")
                            yield Input(
                                value=str(self.ollama_config.connection_config.timeout),
                                placeholder="30",
                                id="timeout_input"
                            )

                            yield Label("Max Retries:")
                            yield Input(
                                value=str(self.ollama_config.connection_config.max_retries),
                                placeholder="3",
                                id="retries_input"
                            )

                # Models Tab - Direct widget yield
                with TabPane("Models", id="models_tab"):
                    yield ModelSelector(on_model_change=self._on_model_selected)

                # Settings Tab - Direct widget yields
                with TabPane("Settings", id="settings_tab"):
                    with Vertical():
                        with Vertical(classes="config-section"):
                            yield Label("Application Settings:")

                            with Horizontal():
                                yield Label("Auto-save chat history:")
                                yield Switch(
                                    value=self.ollama_config.settings.settings.auto_save_chat,
                                    id="auto_save_switch"
                                )

                            yield Label("Max history size:")
                            yield Input(
                                value=str(self.ollama_config.settings.settings.max_history_size),
                                placeholder="1000",
                                id="history_size_input"
                            )

                            yield Label("Theme:")
                            yield Input(
                                value=self.ollama_config.settings.settings.theme,
                                placeholder="dark",
                                id="theme_input"
                            )

            with Horizontal(classes="dialog-buttons"):
                yield Button("Test Connection", id="test_btn", variant="default")
                yield Button("Save", id="save_btn", variant="success")
                yield Button("Cancel", id="cancel_btn", variant="error")

    async def on_mount(self) -> None:
        """Initialize dialog when mounted."""
        await self._update_connection_status()

    async def _update_connection_status(self) -> None:
        """Update the connection status display."""
        status_widget = self.query_one("#connection_status", Static)

        current_status = self.ollama_config.connection_status

        if current_status == ConnectionStatus.CONNECTED:
            status_widget.update("🟢 Connected to Ollama server")
            status_widget.remove_class("disconnected", "connecting")
            status_widget.add_class("connected")
        elif current_status == ConnectionStatus.CONNECTING:
            status_widget.update("🟡 Connecting to Ollama server...")
            status_widget.remove_class("connected", "disconnected")
            status_widget.add_class("connecting")
        elif current_status == ConnectionStatus.ERROR:
            status_widget.update("🔴 Connection failed")
            status_widget.remove_class("connected", "connecting")
            status_widget.add_class("disconnected")
        else:
            status_widget.update("⚪ Not connected")
            status_widget.remove_class("connected", "connecting")
            status_widget.add_class("disconnected")

    async def _on_model_selected(self, model_name: str) -> None:
        """Handle model selection from ModelSelector."""
        if self.on_model_changed:
            await self.on_model_changed(model_name)

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "test_btn":
            await self._test_connection()
        elif event.button.id == "save_btn":
            await self._save_settings()
            self.dismiss(True)
        elif event.button.id == "cancel_btn":
            self.dismiss(False)

    async def _test_connection(self) -> None:
        """Test connection to Ollama server."""
        if self._testing_connection:
            return

        self._testing_connection = True
        test_btn = self.query_one("#test_btn", Button)
        test_btn.label = "Testing..."
        test_btn.disabled = True

        try:
            # Update config with current values
            base_url_input = self.query_one("#base_url_input", Input)
            timeout_input = self.query_one("#timeout_input", Input)
            retries_input = self.query_one("#retries_input", Input)

            self.ollama_config.base_url = base_url_input.value
            self.ollama_config.connection_config.timeout = int(timeout_input.value or "30")
            self.ollama_config.connection_config.max_retries = int(retries_input.value or "3")

            # Test connection
            await self.ollama_config.test_connection()
            await self._update_connection_status()

        except Exception as e:
            status_widget = self.query_one("#connection_status", Static)
            status_widget.update(f"🔴 Connection failed: {str(e)}")
            status_widget.remove_class("connected", "connecting")
            status_widget.add_class("disconnected")
        finally:
            test_btn.label = "Test Connection"
            test_btn.disabled = False
            self._testing_connection = False

    async def _save_settings(self) -> None:
        """Save current settings."""
        try:
            # Connection settings
            base_url_input = self.query_one("#base_url_input", Input)
            timeout_input = self.query_one("#timeout_input", Input)
            retries_input = self.query_one("#retries_input", Input)

            self.ollama_config.base_url = base_url_input.value
            self.ollama_config.connection_config.timeout = int(timeout_input.value or "30")
            self.ollama_config.connection_config.max_retries = int(retries_input.value or "3")

            # General settings
            auto_save_switch = self.query_one("#auto_save_switch", Switch)
            history_size_input = self.query_one("#history_size_input", Input)
            theme_input = self.query_one("#theme_input", Input)

            self.ollama_config.settings.settings.auto_save_chat = auto_save_switch.value
            self.ollama_config.settings.settings.max_history_size = int(history_size_input.value or "1000")
            self.ollama_config.settings.settings.theme = theme_input.value or "dark"

            # Save to file
            await self.ollama_config.save()

        except Exception as e:
            # Handle save error - could show a notification
            pass

    async def on_key(self, event) -> None:
        """Handle key presses."""
        if event.key == "escape":
            self.dismiss(None)
        elif event.key == "ctrl+s":
            asyncio.create_task(self._save_settings())