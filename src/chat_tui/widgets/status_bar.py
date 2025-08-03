import asyncio
from typing import Optional, Dict, Any

from textual.app import ComposeResult
from textual.widgets import Static
from textual.reactive import reactive
from textual.containers import Horizontal

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from ollama import get_ollama_config, get_model_manager
from shared.types import ConnectionStatus
from shared.utils import format_size


class StatusBar(Horizontal):
    """Status bar showing connection status, current model, and system info."""

    DEFAULT_CSS = """
    StatusBar {
        dock: bottom;
        height: 1;
        background: $primary;
        color: $text;
    }

    .status-item {
        height: 1;
        padding: 0 1;
        background: $primary;
        color: $text;
    }

    .status-connection {
        background: $primary;
    }

    .status-connection.connected {
        background: $success;
        color: $text;
    }

    .status-connection.disconnected {
        background: $error;
        color: $text;
    }

    .status-connection.connecting {
        background: $warning;
        color: $text;
    }

    .status-model {
        background: $accent;
        color: $text;
    }

    .status-memory {
        background: $surface;
        color: $text;
    }

    .status-spacer {
        background: $primary;
        color: $text;
    }
    """

    connection_status: reactive[ConnectionStatus] = reactive(ConnectionStatus.DISCONNECTED)
    current_model: reactive[Optional[str]] = reactive(None)
    memory_info: reactive[Optional[Dict[str, Any]]] = reactive(None)

    def __init__(self):
        super().__init__()
        self.ollama_config = get_ollama_config()
        self.model_manager = get_model_manager()
        self._update_task: Optional[asyncio.Task] = None

    def compose(self) -> ComposeResult:
        """Compose the status bar - Fixed to properly yield widgets."""
        yield Static("⚪ Disconnected", id="connection_status", classes="status-item status-connection disconnected")
        yield Static("📦 No Model", id="model_status", classes="status-item status-model")
        yield Static("", id="spacer", classes="status-item status-spacer")  # Spacer to push items
        yield Static("💾 Memory: 0GB", id="memory_status", classes="status-item status-memory")
        yield Static("⌨️ Ctrl+O: Options | Ctrl+Q: Quit", id="shortcuts", classes="status-item")

    async def on_mount(self) -> None:
        """Start status updates when mounted."""
        await self.update_status()
        self._start_periodic_updates()

    def on_unmount(self) -> None:
        """Stop updates when unmounted."""
        if self._update_task and not self._update_task.done():
            self._update_task.cancel()

    def _start_periodic_updates(self) -> None:
        """Start periodic status updates."""
        if self._update_task is None or self._update_task.done():
            self._update_task = asyncio.create_task(self._periodic_update())

    async def _periodic_update(self) -> None:
        """Periodically update status information."""
        while True:
            try:
                await asyncio.sleep(5)  # Update every 5 seconds
                await self.update_status()
            except asyncio.CancelledError:
                break
            except Exception:
                # Continue running even if one update fails
                pass

    async def update_status(self) -> None:
        """Update all status information."""
        await self._update_connection_status()
        await self._update_model_status()
        await self._update_memory_status()

    async def _update_connection_status(self) -> None:
        """Update the connection status display."""
        try:
            status_widget = self.query_one("#connection_status", Static)
            current_status = self.ollama_config.connection_status

            if current_status == ConnectionStatus.CONNECTED:
                status_widget.update("🟢 Connected")
                status_widget.remove_class("disconnected", "connecting")
                status_widget.add_class("connected")
            elif current_status == ConnectionStatus.CONNECTING:
                status_widget.update("🟡 Connecting...")
                status_widget.remove_class("connected", "disconnected")
                status_widget.add_class("connecting")
            elif current_status == ConnectionStatus.ERROR:
                status_widget.update("🔴 Error")
                status_widget.remove_class("connected", "connecting")
                status_widget.add_class("disconnected")
            else:
                status_widget.update("⚪ Disconnected")
                status_widget.remove_class("connected", "connecting")
                status_widget.add_class("disconnected")

            self.connection_status = current_status
        except Exception:
            # Widget might not be mounted yet
            pass

    async def _update_model_status(self) -> None:
        """Update the current model display."""
        try:
            model_widget = self.query_one("#model_status", Static)
            current_model = self.model_manager.get_current_model()

            if current_model:
                model_widget.update(f"📦 {current_model}")
                self.current_model = current_model
            else:
                model_widget.update("📦 No Model")
                self.current_model = None
        except Exception:
            # Widget might not be mounted yet
            pass

    async def _update_memory_status(self) -> None:
        """Update memory usage information."""
        try:
            memory_widget = self.query_one("#memory_status", Static)

            # Get memory info from system if available
            try:
                import psutil
                memory = psutil.virtual_memory()
                used_gb = memory.used / (1024 ** 3)
                total_gb = memory.total / (1024 ** 3)
                memory_widget.update(f"💾 {used_gb:.1f}GB/{total_gb:.1f}GB")

                self.memory_info = {
                    "used": memory.used,
                    "total": memory.total,
                    "percent": memory.percent
                }
            except ImportError:
                # psutil not available, show basic info
                memory_widget.update("💾 Memory: N/A")
                self.memory_info = None
        except Exception:
            # Widget might not be mounted yet
            pass

    async def set_temporary_message(self, message: str, duration: float = 3.0) -> None:
        """Show a temporary message in the spacer area."""
        try:
            # Find a suitable widget to show the message
            spacer_widget = self.query_one("#spacer", Static)
            original_content = spacer_widget.renderable

            # Show temporary message
            spacer_widget.update(f"ℹ️ {message}")

            # Restore original content after duration
            async def restore():
                await asyncio.sleep(duration)
                spacer_widget.update(original_content)

            asyncio.create_task(restore())
        except Exception:
            # Widget might not be mounted yet
            pass

    async def show_model_changed(self, model_name: str) -> None:
        """Show model changed notification."""
        await self.set_temporary_message(f"Model changed to {model_name}")
        await self._update_model_status()

    async def show_connection_changed(self) -> None:
        """Show connection status changed notification."""
        await self._update_connection_status()

        status = self.ollama_config.connection_status
        if status == ConnectionStatus.CONNECTED:
            await self.set_temporary_message("Connected to Ollama")
        elif status == ConnectionStatus.ERROR:
            await self.set_temporary_message("Connection failed")

    def update_shortcuts_display(self, shortcuts: str) -> None:
        """Update the shortcuts display."""
        try:
            shortcuts_widget = self.query_one("#shortcuts", Static)
            shortcuts_widget.update(shortcuts)
        except Exception:
            # Widget might not be mounted yet
            pass