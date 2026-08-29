import json
import logging
import logging.config
from logging.handlers import RotatingFileHandler
from pathlib import Path
from src.config.files_and_directories import APP_LOG_FILE, PROMPT_LOG_FILE


DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


# ======================================================
# APPLICATION LOGS
# ======================================================

class AppLogFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "line": record.lineno,
            "message": record.getMessage(),
        }

        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_data)


_app_log_config = RotatingFileHandler(
    APP_LOG_FILE,
    maxBytes=5_000_000, # Caps at 5 MB
    backupCount=1, # Up to 1 historical log files
    encoding="utf-8"
)
_app_log_config.setFormatter(AppLogFormatter(datefmt=DATE_FORMAT))
_app_log_config.setLevel(logging.INFO)


def app_logger(name: str) -> logging.Logger:
    """
    Returns a module logger configured to write
    json to '~/.agent_app/logs/app.log'.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Avoid adding duplicate handlers if get_logger is called multiple times
    if not logger.handlers:
        logger.addHandler(_app_log_config)
        logger.propagate = False # Keep logs isolated to the log file

    return logger


# ======================================================
# PROMPTS LOGS
# ======================================================

class PromptLogFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        timestamp = self.formatTime(record, self.datefmt)
        info = f"[{timestamp}] [{record.levelname}] [{record.name}:{record.lineno}]"
        prompts = record.getMessage()

        # Formating for multi-line text
        log_entry = f"{info}\n{prompts}\n"

        if record.exc_info:
            exc_text = self.formatException(record.exc_info)
            log_entry += f"\nException:\n{exc_text}\n"

        # Divider line
        log_entry += "=" * 100 + "\n"

        return log_entry


_prompt_log_config = RotatingFileHandler(
    PROMPT_LOG_FILE,
    maxBytes=500_000_000, # Caps at 500 MB
    backupCount=1, # Up to 1 historical log files
    encoding="utf-8"
)
_prompt_log_config.setFormatter(PromptLogFormatter(datefmt=DATE_FORMAT))
_prompt_log_config.setLevel(logging.INFO)


def prompt_logger(name: str) -> logging.Logger:
    """
    Returns a module logger configured to write structured
    plain text to '~/.agent_app/logs/prompts.log'.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Avoid adding duplicate handlers if get_logger is called multiple times
    if not logger.handlers:
        logger.addHandler(_prompt_log_config)
        logger.propagate = False # Keep logs isolated to the log file

    return logger
