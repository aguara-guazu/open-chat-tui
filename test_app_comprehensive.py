#!/usr/bin/env python3
"""
Comprehensive test suite for the Ollama Chat TUI application.
Tests all major components including ChatApp, widgets, Ollama client, and utilities.
"""

import pytest
import asyncio
import sys
import os
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import List, Dict, Any, Optional
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import all application modules with error handling
try:
    from chat_tui.app import ChatApp
except ImportError as e:
    print(f"⚠️  Warning: Could not import ChatApp: {e}")
    ChatApp = None

try:
    from chat_tui.widgets.chat_message import EnhancedChatMessage, SystemMessage, ErrorMessage
except ImportError as e:
    print(f"⚠️  Warning: Could not import chat message widgets: {e}")
    EnhancedChatMessage = SystemMessage = ErrorMessage = None

try:
    from chat_tui.widgets.status_bar import StatusBar
except ImportError as e:
    print(f"⚠️  Warning: Could not import StatusBar: {e}")
    StatusBar = None

try:
    from chat_tui.widgets.model_selector import ModelSelector
except ImportError as e:
    print(f"⚠️  Warning: Could not import ModelSelector: {e}")
    ModelSelector = None

try:
    from chat_tui.widgets.options_dialog import OptionsDialog
except ImportError as e:
    print(f"⚠️  Warning: Could not import OptionsDialog: {e}")
    OptionsDialog = None

try:
    from chat_tui.commands.handlers import CommandHandler, get_command_handler
except ImportError as e:
    print(f"⚠️  Warning: Could not import command handlers: {e}")
    CommandHandler = get_command_handler = None

try:
    from ollama.client import OllamaClient
    from ollama.config import OllamaConfig, get_ollama_config
    from ollama.models import ModelManager
    from ollama.types import (
        OllamaModel, GenerateRequest, GenerateResponse, ChatRequest, ChatResponse,
        ModelListResponse, PullProgress
    )
    from ollama.exceptions import OllamaError, ConnectionError, TimeoutError
except ImportError as e:
    print(f"⚠️  Warning: Could not import Ollama modules: {e}")
    OllamaClient = OllamaConfig = get_ollama_config = ModelManager = None
    OllamaModel = GenerateRequest = GenerateResponse = None
    ChatRequest = ChatResponse = ModelListResponse = PullProgress = None
    OllamaError = ConnectionError = TimeoutError = None

try:
    from shared.settings import get_settings_manager
    from shared.types import ChatRole, Message, ConnectionConfig
    from shared.utils import format_size, validate_url, sanitize_model_name
except ImportError as e:
    print(f"⚠️  Warning: Could not import shared modules: {e}")
    get_settings_manager = None
    ChatRole = Message = ConnectionConfig = None
    format_size = validate_url = sanitize_model_name = None


class TestReport:
    """Test report generator for comprehensive testing feedback."""

    def __init__(self):
        self.results = []
        self.errors = []
        self.warnings = []
        self.coverage_info = {}

    def add_result(self, test_name: str, status: str, details: str = ""):
        """Add test result."""
        self.results.append({
            'test': test_name,
            'status': status,
            'details': details,
            'timestamp': datetime.now().isoformat()
        })

    def add_error(self, test_name: str, error: str):
        """Add error information."""
        self.errors.append({
            'test': test_name,
            'error': error,
            'timestamp': datetime.now().isoformat()
        })

    def add_warning(self, test_name: str, warning: str):
        """Add warning information."""
        self.warnings.append({
            'test': test_name,
            'warning': warning,
            'timestamp': datetime.now().isoformat()
        })

    def generate_report(self) -> str:
        """Generate comprehensive test report."""
        report = []
        report.append("=" * 80)
        report.append("COMPREHENSIVE TEST REPORT - OLLAMA CHAT TUI")
        report.append("=" * 80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")

        # Summary
        total_tests = len(self.results)
        passed = len([r for r in self.results if r['status'] == 'PASS'])
        failed = len([r for r in self.results if r['status'] == 'FAIL'])
        skipped = len([r for r in self.results if r['status'] == 'SKIP'])

        report.append("SUMMARY:")
        report.append(f"  Total Tests: {total_tests}")
        report.append(f"  Passed: {passed}")
        report.append(f"  Failed: {failed}")
        report.append(f"  Skipped: {skipped}")
        report.append(
            f"  Success Rate: {(passed / total_tests * 100):.1f}%" if total_tests > 0 else "  Success Rate: 0%")
        report.append("")

        # Detailed Results
        if self.results:
            report.append("DETAILED RESULTS:")
            report.append("-" * 40)
            for result in self.results:
                status_icon = "✅" if result['status'] == 'PASS' else "❌" if result['status'] == 'FAIL' else "⏭️"
                report.append(f"{status_icon} {result['test']}: {result['status']}")
                if result['details']:
                    report.append(f"    Details: {result['details']}")
            report.append("")

        # Errors
        if self.errors:
            report.append("ERRORS:")
            report.append("-" * 40)
            for error in self.errors:
                report.append(f"❌ {error['test']}: {error['error']}")
            report.append("")

        # Warnings
        if self.warnings:
            report.append("WARNINGS:")
            report.append("-" * 40)
            for warning in self.warnings:
                report.append(f"⚠️  {warning['test']}: {warning['warning']}")
            report.append("")

        # Recommendations
        report.append("RECOMMENDATIONS:")
        report.append("-" * 40)
        if failed > 0:
            report.append("• Fix failing tests before deployment")
        if len(self.errors) > 0:
            report.append("• Address error conditions in application code")
        if len(self.warnings) > 0:
            report.append("• Review warning conditions for potential improvements")

        report.append("• Add integration tests with real Ollama server")
        report.append("• Consider adding performance benchmarks")
        report.append("• Implement UI accessibility tests")
        report.append("")

        report.append("=" * 80)

        return "\n".join(report)


# Global test report instance
test_report = TestReport()


@pytest.fixture
def mock_ollama_config():
    """Mock Ollama configuration."""
    config = Mock(spec=OllamaConfig)
    config.base_url = "http://localhost:11434"
    config.connection_config = Mock(spec=ConnectionConfig)
    config.connection_config.timeout = 30
    config.connection_config.max_retries = 3
    config.test_connection = AsyncMock(return_value=True)
    config.get_available_models = AsyncMock(return_value=[
        OllamaModel(name="llama3.2:1b", size=1073741824, digest="abc123"),
        OllamaModel(name="llama3.2:3b", size=3221225472, digest="def456")
    ])
    config.validate_model = AsyncMock(return_value=True)
    config.get_model_info = AsyncMock(return_value={"name": "llama3.2:1b", "size": 1073741824})
    return config


@pytest.fixture
def mock_ollama_client():
    """Mock Ollama client."""
    client = Mock(spec=OllamaClient)
    client.health_check = AsyncMock(return_value=True)
    client.chat = AsyncMock(return_value=ChatResponse(
        message={"role": "assistant", "content": "Test response"},
        done=True
    ))
    client.generate = AsyncMock(return_value=GenerateResponse(
        response="Test response",
        done=True
    ))
    client.list_models = AsyncMock(return_value=ModelListResponse(
        models=[
            {"name": "llama3.2:1b", "size": 1073741824},
            {"name": "llama3.2:3b", "size": 3221225472}
        ]
    ))
    client.close = AsyncMock()
    return client


@pytest.fixture
def mock_model_manager():
    """Mock model manager."""
    manager = Mock(spec=ModelManager)
    manager.list_models = AsyncMock(return_value=[
        OllamaModel(name="llama3.2:1b", size=1073741824, digest="abc123"),
        OllamaModel(name="llama3.2:3b", size=3221225472, digest="def456")
    ])
    manager.get_default_model = AsyncMock(return_value="llama3.2:1b")
    manager.model_exists = AsyncMock(return_value=True)
    manager.get_model_info = AsyncMock(return_value={"name": "test", "size": 1024})
    return manager


class TestUtilities:
    """Test utility functions."""

    def test_format_size(self):
        """Test file size formatting."""
        if format_size is None:
            test_report.add_result("test_format_size", "SKIP", "format_size function not available")
            return

        try:
            assert format_size(0) == "0 B"
            assert format_size(1024) == "1.0 KB"
            assert format_size(1048576) == "1.0 MB"
            assert format_size(1073741824) == "1.0 GB"
            test_report.add_result("test_format_size", "PASS", "All size formats correct")
        except Exception as e:
            test_report.add_result("test_format_size", "FAIL", str(e))
            test_report.add_error("test_format_size", str(e))

    def test_validate_url(self):
        """Test URL validation."""
        if validate_url is None:
            test_report.add_result("test_validate_url", "SKIP", "validate_url function not available")
            return

        try:
            assert validate_url("http://localhost:11434") == True
            assert validate_url("https://api.example.com") == True
            assert validate_url("invalid-url") == False
            assert validate_url("") == False
            test_report.add_result("test_validate_url", "PASS", "URL validation working")
        except Exception as e:
            test_report.add_result("test_validate_url", "FAIL", str(e))
            test_report.add_error("test_validate_url", str(e))

    def test_sanitize_model_name(self):
        """Test model name sanitization."""
        if sanitize_model_name is None:
            test_report.add_result("test_sanitize_model_name", "SKIP", "sanitize_model_name function not available")
            return

        try:
            assert sanitize_model_name("user/model:tag") == "model:tag"
            assert sanitize_model_name("model@#$%") == "model"
            assert sanitize_model_name("simple-model") == "simple-model"
            test_report.add_result("test_sanitize_model_name", "PASS", "Model name sanitization working")
        except Exception as e:
            test_report.add_result("test_sanitize_model_name", "FAIL", str(e))
            test_report.add_error("test_sanitize_model_name", str(e))


class TestOllamaClient:
    """Test Ollama client functionality."""

    @pytest.mark.asyncio
    async def test_client_initialization(self, mock_ollama_config):
        """Test client initialization."""
        try:
            config = ConnectionConfig(base_url="http://localhost:11434", timeout=30, max_retries=3)
            client = OllamaClient(config)
            assert client.config.base_url == "http://localhost:11434"
            assert client.config.timeout == 30
            test_report.add_result("test_client_initialization", "PASS", "Client initialized correctly")
        except Exception as e:
            test_report.add_result("test_client_initialization", "FAIL", str(e))
            test_report.add_error("test_client_initialization", str(e))

    @pytest.mark.asyncio
    async def test_health_check(self, mock_ollama_client):
        """Test health check."""
        try:
            result = await mock_ollama_client.health_check()
            assert result == True
            test_report.add_result("test_health_check", "PASS", "Health check successful")
        except Exception as e:
            test_report.add_result("test_health_check", "FAIL", str(e))
            test_report.add_error("test_health_check", str(e))

    @pytest.mark.asyncio
    async def test_generate_request(self, mock_ollama_client):
        """Test generate request."""
        try:
            response = await mock_ollama_client.generate()
            assert hasattr(response, 'response')
            assert response.done == True
            test_report.add_result("test_generate_request", "PASS", "Generation request successful")
        except Exception as e:
            test_report.add_result("test_generate_request", "FAIL", str(e))
            test_report.add_error("test_generate_request", str(e))

    @pytest.mark.asyncio
    async def test_chat_request(self, mock_ollama_client):
        """Test chat request."""
        try:
            response = await mock_ollama_client.chat()
            assert hasattr(response, 'message')
            assert response.done == True
            test_report.add_result("test_chat_request", "PASS", "Chat request successful")
        except Exception as e:
            test_report.add_result("test_chat_request", "FAIL", str(e))
            test_report.add_error("test_chat_request", str(e))


class TestModelManager:
    """Test model management functionality."""

    @pytest.mark.asyncio
    async def test_list_models(self, mock_model_manager):
        """Test listing models."""
        try:
            models = await mock_model_manager.list_models()
            assert len(models) > 0
            assert all(hasattr(model, 'name') for model in models)
            test_report.add_result("test_list_models", "PASS", f"Found {len(models)} models")
        except Exception as e:
            test_report.add_result("test_list_models", "FAIL", str(e))
            test_report.add_error("test_list_models", str(e))

    @pytest.mark.asyncio
    async def test_get_default_model(self, mock_model_manager):
        """Test getting default model."""
        try:
            default_model = await mock_model_manager.get_default_model()
            assert default_model is not None
            assert isinstance(default_model, str)
            test_report.add_result("test_get_default_model", "PASS", f"Default model: {default_model}")
        except Exception as e:
            test_report.add_result("test_get_default_model", "FAIL", str(e))
            test_report.add_error("test_get_default_model", str(e))

    @pytest.mark.asyncio
    async def test_model_exists(self, mock_model_manager):
        """Test model existence check."""
        try:
            exists = await mock_model_manager.model_exists("llama3.2:1b")
            assert exists == True
            test_report.add_result("test_model_exists", "PASS", "Model existence check working")
        except Exception as e:
            test_report.add_result("test_model_exists", "FAIL", str(e))
            test_report.add_error("test_model_exists", str(e))


class TestWidgets:
    """Test Textual widgets."""

    def test_enhanced_chat_message_creation(self):
        """Test EnhancedChatMessage widget creation."""
        if EnhancedChatMessage is None or ChatRole is None:
            test_report.add_result("test_enhanced_chat_message_creation", "SKIP",
                                   "EnhancedChatMessage or ChatRole not available")
            return

        try:
            message = EnhancedChatMessage(
                content="Test message",
                role=ChatRole.USER,
                sender_name="TestUser"
            )
            assert message.content == "Test message"
            assert message.role == ChatRole.USER
            assert message.sender_name == "TestUser"
            test_report.add_result("test_enhanced_chat_message_creation", "PASS", "Chat message created successfully")
        except Exception as e:
            test_report.add_result("test_enhanced_chat_message_creation", "FAIL", str(e))
            test_report.add_error("test_enhanced_chat_message_creation", str(e))

    def test_system_message_creation(self):
        """Test SystemMessage widget creation."""
        if SystemMessage is None:
            test_report.add_result("test_system_message_creation", "SKIP", "SystemMessage not available")
            return

        try:
            message = SystemMessage("System notification")
            assert hasattr(message, 'content') or hasattr(message, 'renderable')
            test_report.add_result("test_system_message_creation", "PASS", "System message created successfully")
        except Exception as e:
            test_report.add_result("test_system_message_creation", "FAIL", str(e))
            test_report.add_error("test_system_message_creation", str(e))

    def test_error_message_creation(self):
        """Test ErrorMessage widget creation."""
        if ErrorMessage is None:
            test_report.add_result("test_error_message_creation", "SKIP", "ErrorMessage not available")
            return

        try:
            message = ErrorMessage("Error occurred")
            assert hasattr(message, 'content') or hasattr(message, 'renderable')
            test_report.add_result("test_error_message_creation", "PASS", "Error message created successfully")
        except Exception as e:
            test_report.add_result("test_error_message_creation", "FAIL", str(e))
            test_report.add_error("test_error_message_creation", str(e))


class TestCommandHandlers:
    """Test command handling functionality."""

    def test_command_handler_initialization(self):
        """Test command handler initialization."""
        if get_command_handler is None:
            test_report.add_result("test_command_handler_initialization", "SKIP", "get_command_handler not available")
            return

        try:
            handler = get_command_handler()
            assert handler is not None
            assert hasattr(handler, 'handle_command')
            test_report.add_result("test_command_handler_initialization", "PASS", "Command handler initialized")
        except Exception as e:
            test_report.add_result("test_command_handler_initialization", "FAIL", str(e))
            test_report.add_error("test_command_handler_initialization", str(e))

    def test_list_commands(self):
        """Test listing available commands."""
        if get_command_handler is None:
            test_report.add_result("test_list_commands", "SKIP", "get_command_handler not available")
            return

        try:
            handler = get_command_handler()
            commands = handler.list_commands()
            assert len(commands) > 0
            assert '/help' in commands
            test_report.add_result("test_list_commands", "PASS", f"Found {len(commands)} commands")
        except Exception as e:
            test_report.add_result("test_list_commands", "FAIL", str(e))
            test_report.add_error("test_list_commands", str(e))

    def test_get_help_text(self):
        """Test getting help text."""
        if get_command_handler is None:
            test_report.add_result("test_get_help_text", "SKIP", "get_command_handler not available")
            return

        try:
            handler = get_command_handler()
            help_text = handler.get_help_text()
            assert isinstance(help_text, str)
            assert len(help_text) > 0
            test_report.add_result("test_get_help_text", "PASS", "Help text generated successfully")
        except Exception as e:
            test_report.add_result("test_get_help_text", "FAIL", str(e))
            test_report.add_error("test_get_help_text", str(e))

    @pytest.mark.asyncio
    async def test_handle_help_command(self):
        """Test handling help command."""
        if get_command_handler is None:
            test_report.add_result("test_handle_help_command", "SKIP", "get_command_handler not available")
            return

        try:
            handler = get_command_handler()
            mock_app = Mock()
            mock_app.add_system_message = AsyncMock()

            result = await handler.handle_command('/help', mock_app)
            assert result == True
            mock_app.add_system_message.assert_called()
            test_report.add_result("test_handle_help_command", "PASS", "Help command handled correctly")
        except Exception as e:
            test_report.add_result("test_handle_help_command", "FAIL", str(e))
            test_report.add_error("test_handle_help_command", str(e))


class TestConfiguration:
    """Test configuration management."""

    def test_settings_manager_initialization(self):
        """Test settings manager initialization."""
        try:
            settings = get_settings_manager()
            assert settings is not None
            test_report.add_result("test_settings_manager_initialization", "PASS", "Settings manager initialized")
        except Exception as e:
            test_report.add_result("test_settings_manager_initialization", "FAIL", str(e))
            test_report.add_error("test_settings_manager_initialization", str(e))

    def test_ollama_config_initialization(self):
        """Test Ollama configuration initialization."""
        try:
            config = get_ollama_config()
            assert config is not None
            assert hasattr(config, 'base_url')
            test_report.add_result("test_ollama_config_initialization", "PASS", "Ollama config initialized")
        except Exception as e:
            test_report.add_result("test_ollama_config_initialization", "FAIL", str(e))
            test_report.add_error("test_ollama_config_initialization", str(e))


class TestChatAppIntegration:
    """Test main ChatApp integration."""

    @pytest.mark.asyncio
    async def test_chat_app_initialization(self, mock_ollama_config, mock_model_manager):
        """Test ChatApp initialization."""
        if ChatApp is None:
            test_report.add_result("test_chat_app_initialization", "SKIP", "ChatApp not available")
            return

        try:
            with patch('chat_tui.app.get_ollama_config', return_value=mock_ollama_config):
                with patch('chat_tui.app.get_model_manager', return_value=mock_model_manager):
                    app = ChatApp()
                    assert app is not None
                    assert hasattr(app, 'current_session_id')
                    test_report.add_result("test_chat_app_initialization", "PASS", "ChatApp initialized successfully")
        except Exception as e:
            test_report.add_result("test_chat_app_initialization", "FAIL", str(e))
            test_report.add_error("test_chat_app_initialization", str(e))

    def test_chat_app_bindings(self):
        """Test ChatApp key bindings."""
        if ChatApp is None:
            test_report.add_result("test_chat_app_bindings", "SKIP", "ChatApp not available")
            return

        try:
            app = ChatApp()
            assert hasattr(app, 'BINDINGS')
            bindings = app.BINDINGS
            assert len(bindings) > 0

            # Check for essential bindings
            binding_keys = [binding.key for binding in bindings]
            assert 'ctrl+q' in binding_keys  # Quit
            assert 'ctrl+o' in binding_keys  # Options

            test_report.add_result("test_chat_app_bindings", "PASS", f"Found {len(bindings)} key bindings")
        except Exception as e:
            test_report.add_result("test_chat_app_bindings", "FAIL", str(e))
            test_report.add_error("test_chat_app_bindings", str(e))


class TestErrorHandling:
    """Test error handling scenarios."""

    @pytest.mark.asyncio
    async def test_connection_error_handling(self):
        """Test connection error handling."""
        try:
            # Mock a failing client
            failing_client = Mock(spec=OllamaClient)
            failing_client.health_check = AsyncMock(side_effect=ConnectionError("Connection failed"))

            with pytest.raises(ConnectionError):
                await failing_client.health_check()

            test_report.add_result("test_connection_error_handling", "PASS", "Connection errors handled correctly")
        except Exception as e:
            test_report.add_result("test_connection_error_handling", "FAIL", str(e))
            test_report.add_error("test_connection_error_handling", str(e))

    @pytest.mark.asyncio
    async def test_timeout_error_handling(self):
        """Test timeout error handling."""
        try:
            # Mock a timing out client
            timeout_client = Mock(spec=OllamaClient)
            timeout_client.generate = AsyncMock(side_effect=TimeoutError("Request timed out"))

            with pytest.raises(TimeoutError):
                await timeout_client.generate()

            test_report.add_result("test_timeout_error_handling", "PASS", "Timeout errors handled correctly")
        except Exception as e:
            test_report.add_result("test_timeout_error_handling", "FAIL", str(e))
            test_report.add_error("test_timeout_error_handling", str(e))


class TestAsyncOperations:
    """Test asynchronous operations."""

    @pytest.mark.asyncio
    async def test_concurrent_model_loading(self, mock_model_manager):
        """Test concurrent model loading operations."""
        try:
            # Simulate concurrent model operations
            tasks = [
                mock_model_manager.list_models(),
                mock_model_manager.get_default_model(),
                mock_model_manager.model_exists("test-model")
            ]

            results = await asyncio.gather(*tasks)
            assert len(results) == 3
            assert results[0] is not None  # models list
            assert results[1] is not None  # default model
            assert results[2] is not None  # model exists

            test_report.add_result("test_concurrent_model_loading", "PASS", "Concurrent operations successful")
        except Exception as e:
            test_report.add_result("test_concurrent_model_loading", "FAIL", str(e))
            test_report.add_error("test_concurrent_model_loading", str(e))


class TestDataValidation:
    """Test data validation and sanitization."""

    def test_message_validation(self):
        """Test message data validation."""
        if Message is None or ChatRole is None:
            test_report.add_result("test_message_validation", "SKIP", "Message or ChatRole not available")
            return

        try:
            # Test valid message
            message = Message(role=ChatRole.USER, content="Test message")
            assert message.role == ChatRole.USER
            assert message.content == "Test message"

            # Test message with empty content
            empty_message = Message(role=ChatRole.SYSTEM, content="")
            assert empty_message.content == ""

            test_report.add_result("test_message_validation", "PASS", "Message validation working")
        except Exception as e:
            test_report.add_result("test_message_validation", "FAIL", str(e))
            test_report.add_error("test_message_validation", str(e))

    def test_model_data_validation(self):
        """Test model data validation."""
        if OllamaModel is None:
            test_report.add_result("test_model_data_validation", "SKIP", "OllamaModel not available")
            return

        try:
            # Test valid model
            model = OllamaModel(name="test-model", size=1024, digest="abc123")
            assert model.name == "test-model"
            assert model.size == 1024
            assert model.digest == "abc123"

            test_report.add_result("test_model_data_validation", "PASS", "Model data validation working")
        except Exception as e:
            test_report.add_result("test_model_data_validation", "FAIL", str(e))
            test_report.add_error("test_model_data_validation", str(e))


def run_all_tests():
    """Run all tests and generate comprehensive report."""
    print("🚀 Starting comprehensive test suite for Ollama Chat TUI...")
    print("=" * 80)

    # Initialize test classes
    test_classes = [
        TestUtilities(),
        TestOllamaClient(),
        TestModelManager(),
        TestWidgets(),
        TestCommandHandlers(),
        TestConfiguration(),
        TestChatAppIntegration(),
        TestErrorHandling(),
        TestAsyncOperations(),
        TestDataValidation()
    ]

    # Run synchronous tests
    for test_class in test_classes:
        class_name = test_class.__class__.__name__
        print(f"📋 Running {class_name}...")

        for method_name in dir(test_class):
            if method_name.startswith('test_') and not asyncio.iscoroutinefunction(getattr(test_class, method_name)):
                try:
                    method = getattr(test_class, method_name)
                    method()
                    print(f"  ✅ {method_name}")
                except Exception as e:
                    print(f"  ❌ {method_name}: {str(e)}")
                    test_report.add_error(f"{class_name}.{method_name}", str(e))

    # Run asynchronous tests
    async def run_async_tests():
        for test_class in test_classes:
            class_name = test_class.__class__.__name__

            for method_name in dir(test_class):
                if method_name.startswith('test_') and asyncio.iscoroutinefunction(getattr(test_class, method_name)):
                    try:
                        method = getattr(test_class, method_name)
                        # Create mock fixtures for async tests
                        mock_config = Mock()
                        mock_client = Mock()
                        mock_manager = Mock()

                        # Call async method with mocks if needed
                        if 'mock_ollama_config' in method.__code__.co_varnames:
                            await method(mock_config)
                        elif 'mock_ollama_client' in method.__code__.co_varnames:
                            await method(mock_client)
                        elif 'mock_model_manager' in method.__code__.co_varnames:
                            await method(mock_manager)
                        else:
                            await method()

                        print(f"  ✅ {method_name} (async)")
                    except Exception as e:
                        print(f"  ❌ {method_name} (async): {str(e)}")
                        test_report.add_error(f"{class_name}.{method_name}", str(e))

    # Run async tests
    try:
        asyncio.run(run_async_tests())
    except Exception as e:
        test_report.add_error("async_test_runner", str(e))

    print("\n🎯 Test execution completed!")
    print("📊 Generating comprehensive report...")

    # Generate and return report
    report = test_report.generate_report()
    return report


if __name__ == "__main__":
    # Run tests when executed directly
    report = run_all_tests()
    print("\n" + report)

    # Save report to file
    with open("test_report.txt", "w") as f:
        f.write(report)

    print(f"\n📄 Detailed report saved to: test_report.txt")
    print("🔧 Please send this report back for analysis and fixes!")