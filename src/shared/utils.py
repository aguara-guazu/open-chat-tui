import re
from typing import Optional
from urllib.parse import urlparse


def format_size(size_bytes: int) -> str:
    """Format size in bytes to human-readable format."""
    if size_bytes == 0:
        return "0 B"
    
    units = ["B", "KB", "MB", "GB", "TB"]
    index = 0
    size = float(size_bytes)
    
    while size >= 1024 and index < len(units) - 1:
        size /= 1024
        index += 1
    
    if index == 0:
        return f"{int(size)} {units[index]}"
    else:
        return f"{size:.1f} {units[index]}"


def validate_url(url: str) -> bool:
    """Validate if a URL is properly formatted."""
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except Exception:
        return False


def sanitize_model_name(name: str) -> str:
    """Sanitize model name for display."""
    # Remove repository prefixes and keep only the model name
    if '/' in name:
        name = name.split('/')[-1]
    
    # Remove any special characters except common ones
    name = re.sub(r'[^\w\-\.:]+', '', name)
    
    return name


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """Truncate text to maximum length with suffix."""
    if len(text) <= max_length:
        return text
    
    return text[:max_length - len(suffix)] + suffix


def parse_model_tag(model_name: str) -> tuple[str, Optional[str]]:
    """Parse model name and tag from full model identifier."""
    if ':' in model_name:
        name, tag = model_name.rsplit(':', 1)
        return name, tag
    return model_name, None


def format_timestamp(timestamp) -> str:
    """Format datetime timestamp for display."""
    if timestamp is None:
        return ""
    
    try:
        if hasattr(timestamp, 'strftime'):
            return timestamp.strftime("%H:%M:%S")
        else:
            # Handle string timestamps
            from datetime import datetime
            dt = datetime.fromisoformat(str(timestamp).replace('Z', '+00:00'))
            return dt.strftime("%H:%M:%S")
    except Exception:
        return str(timestamp)


def safe_json_get(data: dict, path: str, default=None):
    """Safely get nested value from JSON-like dict using dot notation."""
    try:
        keys = path.split('.')
        value = data
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value
    except Exception:
        return default