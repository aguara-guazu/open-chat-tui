from typing import Optional, Dict, Any
from datetime import datetime

from textual.widgets import Static
from textual.reactive import reactive

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from shared.types import ChatRole
from shared.utils import format_timestamp


class EnhancedChatMessage(Static):
    """Enhanced chat message widget with role-based styling and metadata."""

    DEFAULT_CSS = """
    EnhancedChatMessage {
        margin: 0 0 1 0;
        padding: 1;
        height: auto;
        background: $surface;
        border-left: thick $accent;
    }

    EnhancedChatMessage.user {
        background: $primary-lighten-1;
        border-left: thick $success;
    }

    EnhancedChatMessage.assistant {
        background: $surface;
        border-left: thick $accent;
    }

    EnhancedChatMessage.system {
        background: $warning-darken-2;
        border-left: thick $warning;
        color: $warning-lighten-3;
    }

    EnhancedChatMessage.error {
        background: $error-darken-2;
        border-left: thick $error;
        color: $error-lighten-3;
    }

    .message-header {
        text-style: bold;
        margin: 0 0 0 0;
    }

    .message-content {
        margin: 0 0 0 0;
        text-wrap: wrap;
    }

    .message-metadata {
        text-style: italic;
        margin: 0 0 0 0;
        text-align: right;
        color: $text-muted;
    }
    """

    role: reactive[ChatRole] = reactive(ChatRole.USER)
    content: reactive[str] = reactive("")
    timestamp: reactive[Optional[datetime]] = reactive(None)

    def __init__(
            self,
            content: str,
            role: ChatRole = ChatRole.USER,
            sender_name: Optional[str] = None,
            timestamp: Optional[datetime] = None,
            metadata: Optional[Dict[str, Any]] = None,
            show_timestamp: bool = True,
            show_metadata: bool = False,
            **kwargs
    ):
        super().__init__(**kwargs)

        self.content = content
        self.role = role
        self.sender_name = sender_name or self._get_default_sender_name(role)
        self.timestamp = timestamp or datetime.now()
        self.metadata = metadata or {}
        self.show_timestamp = show_timestamp
        self.show_metadata = show_metadata

        # Apply role-based CSS class
        self.add_class(role.value)

        # Update the display
        self._update_display()

    def _get_default_sender_name(self, role: ChatRole) -> str:
        """Get default sender name for role."""
        if role == ChatRole.USER:
            return "You"
        elif role == ChatRole.ASSISTANT:
            return "Assistant"
        elif role == ChatRole.SYSTEM:
            return "System"
        else:
            return "Unknown"

    def _update_display(self) -> None:
        """Update the message display."""
        lines = []

        # Message header with sender and timestamp
        header_parts = [f"**{self.sender_name}**"]

        if self.show_timestamp and self.timestamp:
            header_parts.append(f"({format_timestamp(self.timestamp)})")

        header = " ".join(header_parts)
        lines.append(header)

        # Message content
        lines.append(self.content)

        # Metadata (if enabled and available)
        if self.show_metadata and self.metadata:
            metadata_parts = []

            # Show model info if available
            if 'model' in self.metadata:
                metadata_parts.append(f"Model: {self.metadata['model']}")

            # Show token counts if available
            if 'eval_count' in self.metadata:
                metadata_parts.append(f"Tokens: {self.metadata['eval_count']}")

            # Show timing if available
            if 'eval_duration' in self.metadata and self.metadata['eval_duration']:
                # Convert nanoseconds to milliseconds
                duration_ms = self.metadata['eval_duration'] / 1_000_000
                metadata_parts.append(f"Time: {duration_ms:.0f}ms")

            if metadata_parts:
                lines.append("")  # Empty line
                lines.append(f"_{' | '.join(metadata_parts)}_")

        # Update widget content
        self.update("\n".join(lines))

    def update_content(self, content: str) -> None:
        """Update message content."""
        self.content = content
        self._update_display()

    def append_content(self, additional_content: str) -> None:
        """Append content to existing message (useful for streaming)."""
        self.content += additional_content
        self._update_display()

    def set_metadata(self, metadata: Dict[str, Any]) -> None:
        """Update message metadata."""
        self.metadata.update(metadata)
        if self.show_metadata:
            self._update_display()

    def set_error_state(self, error_message: str) -> None:
        """Set message to error state."""
        self.remove_class("user", "assistant", "system")
        self.add_class("error")
        self.role = ChatRole.SYSTEM  # Use system role for errors
        self.sender_name = "Error"
        self.content = error_message
        self._update_display()

    def toggle_metadata_display(self) -> None:
        """Toggle metadata display."""
        self.show_metadata = not self.show_metadata
        self._update_display()

    def get_role_icon(self) -> str:
        """Get emoji icon for message role."""
        if self.role == ChatRole.USER:
            return "👤"
        elif self.role == ChatRole.ASSISTANT:
            return "🤖"
        elif self.role == ChatRole.SYSTEM:
            return "ℹ️"
        else:
            return "❓"

    def get_display_info(self) -> Dict[str, Any]:
        """Get display information for the message."""
        return {
            'role': self.role.value,
            'sender': self.sender_name,
            'content': self.content,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'metadata': self.metadata,
            'character_count': len(self.content),
            'line_count': len(self.content.split('\n'))
        }


class StreamingChatMessage(EnhancedChatMessage):
    """Chat message widget optimized for streaming content."""

    def __init__(self, role: ChatRole = ChatRole.ASSISTANT, sender_name: Optional[str] = None, **kwargs):
        super().__init__(
            content="",  # Start with empty content
            role=role,
            sender_name=sender_name,
            show_metadata=False,  # Don't show metadata during streaming
            **kwargs
        )
        self._streaming = True
        self._stream_buffer = ""

    def stream_chunk(self, chunk: str) -> None:
        """Add a chunk of streamed content."""
        if self._streaming:
            self._stream_buffer += chunk
            self.content = self._stream_buffer
            self._update_display()

    def finish_streaming(self, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Finish streaming and optionally add metadata."""
        self._streaming = False
        if metadata:
            self.set_metadata(metadata)
            self.show_metadata = True
        self._update_display()

    def is_streaming(self) -> bool:
        """Check if message is currently streaming."""
        return self._streaming


class SystemMessage(EnhancedChatMessage):
    """Specialized system message widget."""

    def __init__(self, content: str, **kwargs):
        super().__init__(
            content=content,
            role=ChatRole.SYSTEM,
            sender_name="System",
            show_timestamp=True,
            show_metadata=False,
            **kwargs
        )


class ErrorMessage(EnhancedChatMessage):
    """Specialized error message widget."""

    def __init__(self, error_content: str, **kwargs):
        super().__init__(
            content=error_content,
            role=ChatRole.SYSTEM,
            sender_name="Error",
            show_timestamp=True,
            show_metadata=False,
            **kwargs
        )

        # Apply error styling
        self.remove_class("system")
        self.add_class("error")