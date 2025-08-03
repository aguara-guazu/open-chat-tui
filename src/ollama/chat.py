import asyncio
from typing import List, Optional, Dict, Any, Callable, Awaitable, AsyncIterator
from datetime import datetime
from dataclasses import dataclass, field

from .client import OllamaClient
from .config import get_ollama_config
from .models import get_model_manager
from .types import ChatRequest, ChatResponse, OllamaMessage, GenerateRequest, GenerateResponse
from .exceptions import OllamaError, ModelNotFoundError
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from shared.types import ChatMessage, ChatRole


@dataclass
class ChatSession:
    """Represents an active chat session."""
    session_id: str
    model_name: str
    messages: List[ChatMessage] = field(default_factory=list)
    system_prompt: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_message(self, role: ChatRole, content: str, metadata: Optional[Dict[str, Any]] = None):
        """Add a message to the session."""
        message = ChatMessage(
            role=role,
            content=content,
            metadata=metadata or {}
        )
        self.messages.append(message)
        self.last_activity = datetime.now()
    
    def get_ollama_messages(self) -> List[OllamaMessage]:
        """Convert messages to Ollama format."""
        ollama_messages = []
        
        # Add system message if present
        if self.system_prompt:
            ollama_messages.append(OllamaMessage(
                role="system",
                content=self.system_prompt
            ))
        
        # Add conversation messages
        for msg in self.messages:
            ollama_messages.append(OllamaMessage(
                role=msg.role.value,
                content=msg.content
            ))
        
        return ollama_messages
    
    def clear_messages(self):
        """Clear all messages from the session."""
        self.messages.clear()
        self.last_activity = datetime.now()
    
    def get_message_count(self) -> int:
        """Get total number of messages."""
        return len(self.messages)
    
    def get_last_user_message(self) -> Optional[ChatMessage]:
        """Get the last user message."""
        for msg in reversed(self.messages):
            if msg.role == ChatRole.USER:
                return msg
        return None
    
    def get_last_assistant_message(self) -> Optional[ChatMessage]:
        """Get the last assistant message."""
        for msg in reversed(self.messages):
            if msg.role == ChatRole.ASSISTANT:
                return msg
        return None


class ChatManager:
    """Manages chat operations with Ollama."""
    
    def __init__(self):
        self.config = get_ollama_config()
        self.model_manager = get_model_manager()
        self._active_sessions: Dict[str, ChatSession] = {}
        self._generation_tasks: Dict[str, asyncio.Task] = {}
    
    async def create_session(
        self,
        session_id: str,
        model_name: Optional[str] = None,
        system_prompt: Optional[str] = None
    ) -> ChatSession:
        """Create a new chat session."""
        # Use default model if none specified
        if not model_name:
            model_name = await self.model_manager.get_default_model()
            if not model_name:
                raise ModelNotFoundError("No models available and no default model set")
        
        # Validate model exists
        if not await self.model_manager.model_exists(model_name):
            raise ModelNotFoundError(f"Model {model_name} not found")
        
        # Validate model is suitable for chat
        if not await self.model_manager.validate_model_for_chat(model_name):
            # Still allow it, but log a warning
            pass
        
        session = ChatSession(
            session_id=session_id,
            model_name=model_name,
            system_prompt=system_prompt
        )
        
        self._active_sessions[session_id] = session
        return session
    
    def get_session(self, session_id: str) -> Optional[ChatSession]:
        """Get an existing chat session."""
        return self._active_sessions.get(session_id)
    
    def get_or_create_session(
        self,
        session_id: str,
        model_name: Optional[str] = None,
        system_prompt: Optional[str] = None
    ) -> ChatSession:
        """Get existing session or create new one."""
        session = self.get_session(session_id)
        if session:
            return session
        
        # Create new session (this is a coroutine, so we need to handle it)
        return asyncio.create_task(
            self.create_session(session_id, model_name, system_prompt)
        )
    
    async def send_message(
        self,
        session_id: str,
        message: str,
        stream: bool = False,
        stream_callback: Optional[Callable[[str], Awaitable[None]]] = None
    ) -> str:
        """Send a message and get response."""
        session = self.get_session(session_id)
        if not session:
            raise OllamaError(f"Session {session_id} not found")
        
        # Add user message to session
        session.add_message(ChatRole.USER, message)
        
        try:
            if stream:
                return await self._send_message_stream(session, stream_callback)
            else:
                return await self._send_message_simple(session)
                
        except Exception as e:
            # Remove the user message if response failed
            if session.messages and session.messages[-1].role == ChatRole.USER:
                session.messages.pop()
            raise e
    
    async def _send_message_simple(self, session: ChatSession) -> str:
        """Send message without streaming."""
        client = self.config.get_client()
        
        request = ChatRequest(
            model=session.model_name,
            messages=session.get_ollama_messages(),
            stream=False
        )
        
        response = await client.chat(request)
        
        # Add assistant response to session
        assistant_message = response.message.content
        session.add_message(ChatRole.ASSISTANT, assistant_message, {
            'model': response.model,
            'created_at': response.created_at,
            'total_duration': response.total_duration,
            'eval_count': response.eval_count,
            'eval_duration': response.eval_duration
        })
        
        return assistant_message
    
    async def _send_message_stream(
        self,
        session: ChatSession,
        stream_callback: Optional[Callable[[str], Awaitable[None]]]
    ) -> str:
        """Send message with streaming response."""
        client = self.config.get_client()
        
        request = ChatRequest(
            model=session.model_name,
            messages=session.get_ollama_messages(),
            stream=True
        )
        
        full_response = ""
        
        async for chunk in client.chat_stream(request):
            if chunk.message and chunk.message.content:
                content = chunk.message.content
                full_response += content
                
                if stream_callback:
                    await stream_callback(content)
        
        # Add complete assistant response to session
        session.add_message(ChatRole.ASSISTANT, full_response)
        
        return full_response
    
    async def generate_response(
        self,
        prompt: str,
        model_name: Optional[str] = None,
        system_prompt: Optional[str] = None,
        stream: bool = False,
        stream_callback: Optional[Callable[[str], Awaitable[None]]] = None
    ) -> str:
        """Generate a single response without session context."""
        # Use default model if none specified
        if not model_name:
            model_name = await self.model_manager.get_default_model()
            if not model_name:
                raise ModelNotFoundError("No models available and no default model set")
        
        client = self.config.get_client()
        
        # Build full prompt with system context
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"System: {system_prompt}\n\nUser: {prompt}"
        
        if stream:
            return await self._generate_stream(client, model_name, full_prompt, stream_callback)
        else:
            return await self._generate_simple(client, model_name, full_prompt)
    
    async def _generate_simple(self, client: OllamaClient, model_name: str, prompt: str) -> str:
        """Generate response without streaming."""
        request = GenerateRequest(
            model=model_name,
            prompt=prompt,
            stream=False
        )
        
        response = await client.generate(request)
        return response.response
    
    async def _generate_stream(
        self,
        client: OllamaClient,
        model_name: str,
        prompt: str,
        stream_callback: Optional[Callable[[str], Awaitable[None]]]
    ) -> str:
        """Generate response with streaming."""
        request = GenerateRequest(
            model=model_name,
            prompt=prompt,
            stream=True
        )
        
        full_response = ""
        
        async for chunk in client.generate_stream(request):
            if chunk.response:
                content = chunk.response
                full_response += content
                
                if stream_callback:
                    await stream_callback(content)
        
        return full_response
    
    async def change_session_model(self, session_id: str, new_model_name: str) -> bool:
        """Change the model for an existing session."""
        session = self.get_session(session_id)
        if not session:
            return False
        
        # Validate new model
        if not await self.model_manager.model_exists(new_model_name):
            raise ModelNotFoundError(f"Model {new_model_name} not found")
        
        session.model_name = new_model_name
        session.last_activity = datetime.now()
        return True
    
    def clear_session(self, session_id: str) -> bool:
        """Clear messages from a session."""
        session = self.get_session(session_id)
        if not session:
            return False
        
        session.clear_messages()
        return True
    
    def delete_session(self, session_id: str) -> bool:
        """Delete a session completely."""
        if session_id in self._active_sessions:
            # Cancel any ongoing generation
            if session_id in self._generation_tasks:
                task = self._generation_tasks[session_id]
                if not task.done():
                    task.cancel()
                del self._generation_tasks[session_id]
            
            del self._active_sessions[session_id]
            return True
        return False
    
    def list_sessions(self) -> List[ChatSession]:
        """List all active sessions."""
        return list(self._active_sessions.values())
    
    def get_session_summary(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get summary information about a session."""
        session = self.get_session(session_id)
        if not session:
            return None
        
        return {
            'session_id': session.session_id,
            'model_name': session.model_name,
            'message_count': session.get_message_count(),
            'created_at': session.created_at.isoformat(),
            'last_activity': session.last_activity.isoformat(),
            'has_system_prompt': session.system_prompt is not None,
            'last_user_message': session.get_last_user_message().content if session.get_last_user_message() else None
        }
    
    async def estimate_context_usage(self, session_id: str) -> Dict[str, Any]:
        """Estimate token usage for a session (rough approximation)."""
        session = self.get_session(session_id)
        if not session:
            return {'error': 'Session not found'}
        
        # Rough token estimation (4 chars ≈ 1 token)
        total_chars = 0
        
        if session.system_prompt:
            total_chars += len(session.system_prompt)
        
        for msg in session.messages:
            total_chars += len(msg.content)
        
        estimated_tokens = total_chars // 4
        
        return {
            'estimated_tokens': estimated_tokens,
            'total_characters': total_chars,
            'message_count': len(session.messages),
            'has_system_prompt': session.system_prompt is not None
        }
    
    async def cleanup(self):
        """Cancel all operations and cleanup sessions."""
        # Cancel all generation tasks
        for task in self._generation_tasks.values():
            if not task.done():
                task.cancel()
        
        if self._generation_tasks:
            await asyncio.gather(*self._generation_tasks.values(), return_exceptions=True)
        
        self._generation_tasks.clear()
        self._active_sessions.clear()


# Global chat manager instance
_chat_manager_instance: Optional[ChatManager] = None


def get_chat_manager() -> ChatManager:
    """Get the global chat manager instance."""
    global _chat_manager_instance
    if _chat_manager_instance is None:
        _chat_manager_instance = ChatManager()
    return _chat_manager_instance


async def cleanup_chat_manager():
    """Cleanup the global chat manager instance."""
    global _chat_manager_instance
    if _chat_manager_instance:
        await _chat_manager_instance.cleanup()
        _chat_manager_instance = None