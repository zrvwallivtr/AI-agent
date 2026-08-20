import json
import logging
import logging.config
from logging.handlers import RotatingFileHandler
from pathlib import Path
from src import config


LOG_DIR = config.APP_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "agent.log"

DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "line": record.lineno,
            "message": record.getMessage(),
        }

        # Include exception tracebacks if 
        # logger.exception() or exc_info=True
        # is called.
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_data)

_file_handler = RotatingFileHandler(
    LOG_FILE,
    maxBytes=5_000_000, # Caps at 5 MB
    backupCount=3, # Up to 3 historical log files
    encoding="utf-8"
)
_file_handler.setFormatter(JsonFormatter(datefmt=DATE_FORMAT))
_file_handler.setLevel(logging.INFO)


def get_logger(name: str) -> logging.Logger:
    """
    Returns a module logger configured to write structured
    plain text to '~/.agent_app/logs/agent.log'.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Avoid adding duplicate handlers if get_logger is called multiple times
    if not logger.handlers:
        logger.addHandler(_file_handler)
        logger.propagate = False # Keep logs isolated to the log file

    return logger
