#!/usr/bin/env python3
"""
Chat TUI - A simple chat interface built with Textual framework.

Usage:
    python chat_tui.py

Commands:
    - Type messages and press Enter to send
    - Type '/exit' to quit the application
"""

import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from chat_tui import run_chat_app


if __name__ == "__main__":
    try:
        run_chat_app()
    except KeyboardInterrupt:
        print("\nGoodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"Error starting chat app: {e}")
        sys.exit(1)