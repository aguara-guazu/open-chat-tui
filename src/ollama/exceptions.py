class OllamaError(Exception):
    """Base exception for Ollama-related errors."""
    
    def __init__(self, message: str, status_code: int = None, response_data: dict = None):
        self.message = message
        self.status_code = status_code
        self.response_data = response_data or {}
        super().__init__(self.message)


class ConnectionError(OllamaError):
    """Raised when unable to connect to Ollama server."""
    pass


class AuthenticationError(OllamaError):
    """Raised when authentication fails."""
    pass


class ModelNotFoundError(OllamaError):
    """Raised when a requested model is not found."""
    pass


class ModelLoadError(OllamaError):
    """Raised when a model fails to load."""
    pass


class InvalidRequestError(OllamaError):
    """Raised when request parameters are invalid."""
    pass


class ServerError(OllamaError):
    """Raised when the Ollama server returns an error."""
    pass


class TimeoutError(OllamaError):
    """Raised when a request times out."""
    pass


class StreamingError(OllamaError):
    """Raised when streaming response fails."""
    pass


class ConfigurationError(OllamaError):
    """Raised when configuration is invalid."""
    pass


def handle_response_error(status_code: int, response_data: dict = None) -> OllamaError:
    """Convert HTTP status code to appropriate exception."""
    response_data = response_data or {}
    error_message = response_data.get('error', f'HTTP {status_code} error')
    
    if status_code == 401:
        return AuthenticationError(error_message, status_code, response_data)
    elif status_code == 404:
        return ModelNotFoundError(error_message, status_code, response_data)
    elif status_code == 400:
        return InvalidRequestError(error_message, status_code, response_data)
    elif status_code >= 500:
        return ServerError(error_message, status_code, response_data)
    else:
        return OllamaError(error_message, status_code, response_data)