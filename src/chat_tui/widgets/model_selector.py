import asyncio
from typing import List, Optional, Callable, Awaitable

from textual.app import ComposeResult
from textual.containers import Vertical, Horizontal
from textual.widgets import Static, Select, Button, Input, Label
from textual.reactive import reactive
from textual.message import Message

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from ollama import get_model_manager, OllamaModel
from shared.utils import format_size


class ModelSelector(Vertical):
    """Widget for selecting and managing Ollama models."""

    DEFAULT_CSS = """
    ModelSelector {
        height: auto;
        border: solid $accent;
        padding: 1;
    }

    .model-info {
        height: auto;
        background: $surface;
        border: solid $primary;
        padding: 0 1;
        margin: 1 0;
    }

    .model-actions {
        height: auto;
        layout: horizontal;
        margin: 1 0 0 0;
    }

    .pull-section {
        height: auto;
        border: solid $warning;
        padding: 1;
        margin: 1 0;
    }

    .section-title {
        text-style: bold;
        margin: 0 0 1 0;
        color: $accent;  
    }
    """

    class ModelSelected(Message):
        """Message sent when a model is selected."""

        def __init__(self, model_name: str) -> None:
            self.model_name = model_name
            super().__init__()

    class ModelDeleted(Message):
        """Message sent when a model is deleted."""

        def __init__(self, model_name: str) -> None:
            self.model_name = model_name
            super().__init__()

    selected_model: reactive[Optional[str]] = reactive(None)

    def __init__(self, on_model_change: Optional[Callable[[str], Awaitable[None]]] = None):
        super().__init__()
        self.model_manager = get_model_manager()
        self.on_model_change = on_model_change
        self._models: List[OllamaModel] = []
        self._loading = False

    def compose(self) -> ComposeResult:
        """Compose the model selector widget - Fixed to not use redundant Vertical container."""
        # Since this widget extends Vertical, we don't need another Vertical wrapper
        yield Label("Select Model:", classes="section-title")
        yield Select([], id="model_select", allow_blank=False)

        yield Static("", id="model_info", classes="model-info")

        with Horizontal(classes="model-actions"):
            yield Button("Refresh", id="refresh_btn", variant="default")
            yield Button("Delete", id="delete_btn", variant="error", disabled=True)

        with Vertical(classes="pull-section"):
            yield Label("Pull New Model:")
            yield Input(placeholder="Enter model name (e.g., llama3.2:1b)", id="pull_input")
            yield Button("Pull Model", id="pull_btn", variant="success")

    async def on_mount(self) -> None:
        """Initialize the widget when mounted."""
        await self.refresh_models()

    async def refresh_models(self) -> None:
        """Refresh the list of available models."""
        if self._loading:
            return

        self._loading = True

        try:
            # Get models from manager
            models = await self.model_manager.list_models()
            self._models = models

            # Update select widget
            select_widget = self.query_one("#model_select", Select)

            # Create options
            options = []
            for model in models:
                # Format the option display
                size_str = format_size(model.size) if hasattr(model, 'size') and model.size else ""
                display = f"{model.name} ({size_str})" if size_str else model.name
                options.append((display, model.name))

            select_widget.set_options(options)

            # If no model is selected and we have models, select the first one
            if not self.selected_model and models:
                first_model = models[0].name
                select_widget.value = first_model
                self.selected_model = first_model
                await self._update_model_info(first_model)

            # Update delete button state
            delete_btn = self.query_one("#delete_btn", Button)
            delete_btn.disabled = len(models) == 0

        except Exception as e:
            # Show error in model info
            info_widget = self.query_one("#model_info", Static)
            info_widget.update(f"Error loading models: {str(e)}")
        finally:
            self._loading = False

    async def _update_model_info(self, model_name: str) -> None:
        """Update the model information display."""
        try:
            info_widget = self.query_one("#model_info", Static)

            # Find the model
            model = next((m for m in self._models if m.name == model_name), None)
            if not model:
                info_widget.update("Model information not available")
                return

            # Format model information
            info_lines = [f"📦 Model: {model.name}"]

            if hasattr(model, 'size') and model.size:
                info_lines.append(f"💾 Size: {format_size(model.size)}")

            if hasattr(model, 'modified_at') and model.modified_at:
                info_lines.append(f"📅 Modified: {model.modified_at.strftime('%Y-%m-%d %H:%M')}")

            if hasattr(model, 'digest') and model.digest:
                info_lines.append(f"🔑 Digest: {model.digest[:16]}...")

            info_widget.update("\n".join(info_lines))

        except Exception as e:
            info_widget = self.query_one("#model_info", Static)
            info_widget.update(f"Error loading model info: {str(e)}")

    def _update_model_info_empty(self) -> None:
        """Clear the model information display."""
        try:
            info_widget = self.query_one("#model_info", Static)
            info_widget.update("No model selected")
        except Exception:
            pass

    async def on_select_changed(self, event: Select.Changed) -> None:
        """Handle model selection changes."""
        if event.select.id == "model_select" and event.value is not None:
            model_name = str(event.value)
            self.selected_model = model_name
            await self._update_model_info(model_name)

            # Notify parent if callback is set
            if self.on_model_change:
                await self.on_model_change(model_name)

            # Send message
            self.post_message(self.ModelSelected(model_name))

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "refresh_btn":
            await self.refresh_models()
        elif event.button.id == "delete_btn":
            await self._delete_selected_model()
        elif event.button.id == "pull_btn":
            await self._pull_model()

    async def _delete_selected_model(self) -> None:
        """Delete the currently selected model."""
        if not self.selected_model:
            return

        try:
            # Ask for confirmation (in a real app you'd show a dialog)
            await self.model_manager.delete_model(self.selected_model)

            # Refresh the model list
            await self.refresh_models()

            # Send message
            self.post_message(self.ModelDeleted(self.selected_model))

            # Clear selection
            self.selected_model = None
            self._update_model_info_empty()

        except Exception as e:
            # Show error in model info
            info_widget = self.query_one("#model_info", Static)
            info_widget.update(f"Error deleting model: {str(e)}")

    async def _pull_model(self) -> None:
        """Pull a new model."""
        pull_input = self.query_one("#pull_input", Input)
        model_name = pull_input.value.strip()

        if not model_name:
            return

        try:
            # Disable the pull button while pulling
            pull_btn = self.query_one("#pull_btn", Button)
            pull_btn.disabled = True
            pull_btn.label = "Pulling..."

            # Show progress in model info
            info_widget = self.query_one("#model_info", Static)
            info_widget.update(f"Pulling model: {model_name}...")

            # Pull the model
            await self.model_manager.pull_model(model_name)

            # Clear input and refresh
            pull_input.value = ""
            await self.refresh_models()

            # Select the newly pulled model
            await self.set_selected_model(model_name)

        except Exception as e:
            info_widget = self.query_one("#model_info", Static)
            info_widget.update(f"Error pulling model: {str(e)}")
        finally:
            # Re-enable the pull button
            pull_btn = self.query_one("#pull_btn", Button)
            pull_btn.disabled = False
            pull_btn.label = "Pull Model"

    def get_selected_model(self) -> Optional[str]:
        """Get the currently selected model name."""
        return self.selected_model

    async def set_selected_model(self, model_name: Optional[str]) -> None:
        """Set the selected model programmatically."""
        if model_name and any(m.name == model_name for m in self._models):
            try:
                select_widget = self.query_one("#model_select", Select)
                select_widget.value = model_name
                self.selected_model = model_name
                await self._update_model_info(model_name)
            except Exception:
                pass
        else:
            self.selected_model = None
            self._update_model_info_empty()