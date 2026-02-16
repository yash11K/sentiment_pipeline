"""
Centralized Logger with Emoji Support for Review Intelligence Application
"""
import logging
import sys
from typing import Optional
from functools import lru_cache


class EmojiLogFormatter(logging.Formatter):
    """Custom formatter that adds emojis based on log level"""
    
    EMOJI_MAP = {
        logging.DEBUG: "🔍",
        logging.INFO: "ℹ️ ",
        logging.WARNING: "⚠️ ",
        logging.ERROR: "❌",
        logging.CRITICAL: "🔥",
    }
    
    def format(self, record: logging.LogRecord) -> str:
        emoji = self.EMOJI_MAP.get(record.levelno, "📝")
        record.emoji = emoji
        return super().format(record)


class Logger:
    """
    Application logger with emoji support and module-specific contexts.
    
    Usage:
        from utils.logger import get_logger
        logger = get_logger(__name__)
        logger.info("Processing started")
        logger.success("Task completed")
    """
    
    # Custom emojis for specific operations
    EMOJIS = {
        "start": "🚀",
        "success": "✅",
        "complete": "🎉",
        "database": "🗄️ ",
        "api": "🌐",
        "s3": "☁️ ",
        "llm": "🤖",
        "bedrock": "🤖",
        "parse": "📄",
        "enrich": "✨",
        "insight": "💡",
        "chat": "💬",
        "filter": "🔎",
        "export": "📤",
        "import": "📥",
        "batch": "📦",
        "progress": "⏳",
        "skip": "⏭️ ",
        "retry": "🔄",
        "config": "⚙️ ",
        "init": "🏁",
        "shutdown": "🛑",
        "user": "👤",
        "location": "📍",
        "review": "📝",
        "sentiment": "😊",
        "topic": "🏷️ ",
        "warning": "⚠️ ",
        "error": "❌",
        "critical": "🔥",
    }
    
    def __init__(self, name: str, level: int = logging.INFO):
        self._logger = logging.getLogger(name)
        self._logger.setLevel(level)
        
        if not self._logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setLevel(level)
            formatter = EmojiLogFormatter(
                fmt="%(emoji)s [%(name)s] %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            )
            handler.setFormatter(formatter)
            self._logger.addHandler(handler)
            self._logger.propagate = False
    
    def _format_message(self, emoji_key: Optional[str], message: str) -> str:
        """Format message with optional emoji prefix"""
        if emoji_key and emoji_key in self.EMOJIS:
            return f"{self.EMOJIS[emoji_key]} {message}"
        return message
    
    # Standard logging methods
    def debug(self, message: str, emoji: Optional[str] = None):
        self._logger.debug(self._format_message(emoji, message))
    
    def info(self, message: str, emoji: Optional[str] = None):
        self._logger.info(self._format_message(emoji, message))
    
    def warning(self, message: str, emoji: Optional[str] = None):
        self._logger.warning(self._format_message(emoji, message))
    
    def error(self, message: str, emoji: Optional[str] = None):
        self._logger.error(self._format_message(emoji, message))
    
    def critical(self, message: str, emoji: Optional[str] = None):
        self._logger.critical(self._format_message(emoji, message))
    
    # Convenience methods with built-in emojis
    def start(self, message: str):
        self._logger.info(f"{self.EMOJIS['start']} {message}")
    
    def success(self, message: str):
        self._logger.info(f"{self.EMOJIS['success']} {message}")
    
    def complete(self, message: str):
        self._logger.info(f"{self.EMOJIS['complete']} {message}")
    
    def progress(self, message: str):
        self._logger.info(f"{self.EMOJIS['progress']} {message}")
    
    def skip(self, message: str):
        self._logger.info(f"{self.EMOJIS['skip']} {message}")
    
    def retry(self, message: str):
        self._logger.warning(f"{self.EMOJIS['retry']} {message}")
    
    # Domain-specific methods
    def database(self, message: str):
        self._logger.info(f"{self.EMOJIS['database']} {message}")
    
    def api(self, message: str):
        self._logger.info(f"{self.EMOJIS['api']} {message}")
    
    def s3(self, message: str):
        self._logger.info(f"{self.EMOJIS['s3']} {message}")
    
    def llm(self, message: str):
        self._logger.info(f"{self.EMOJIS['llm']} {message}")
    
    def parse(self, message: str):
        self._logger.info(f"{self.EMOJIS['parse']} {message}")
    
    def enrich(self, message: str):
        self._logger.info(f"{self.EMOJIS['enrich']} {message}")
    
    def insight(self, message: str):
        self._logger.info(f"{self.EMOJIS['insight']} {message}")
    
    def chat(self, message: str):
        self._logger.info(f"{self.EMOJIS['chat']} {message}")
    
    def batch(self, message: str):
        self._logger.info(f"{self.EMOJIS['batch']} {message}")
    
    def export(self, message: str):
        self._logger.info(f"{self.EMOJIS['export']} {message}")


@lru_cache(maxsize=None)
def get_logger(name: str, level: int = logging.INFO) -> Logger:
    """
    Get or create a logger instance for the given module name.
    Uses caching to ensure single logger per module.
    
    Args:
        name: Module name (typically __name__)
        level: Logging level (default: INFO)
    
    Returns:
        Logger instance with emoji support
    """
    return Logger(name, level)
