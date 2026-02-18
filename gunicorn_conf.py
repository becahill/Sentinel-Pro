import json
import logging
import os
from datetime import datetime

bind = "0.0.0.0:8000"
workers = int(os.getenv("WEB_CONCURRENCY", "2"))
threads = int(os.getenv("WEB_THREADS", "4"))
timeout = int(os.getenv("WEB_TIMEOUT", "60"))
accesslog = "-"
errorlog = "-"
loglevel = "info"

access_log_format = (
    '{'
    '"timestamp":"%(t)s",'
    '"remote_addr":"%(h)s",'
    '"request":"%(r)s",'
    '"status":%(s)s,'
    '"bytes":%(b)s,'
    '"referer":"%(f)s",'
    '"user_agent":"%(a)s"'
    '}'
)


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload)


def on_starting(server):
    server.log.info("Starting gunicorn server")


def worker_int(worker):
    worker.log.info("Worker received INT or QUIT")


def when_ready(server):
    server.log.info("Server is ready. Spawning workers")


logconfig_dict = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "json": {
            "()": JsonFormatter,
        },
        "plain": {
            "format": "%(message)s",
        },
    },
    "handlers": {
        "stdout": {
            "class": "logging.StreamHandler",
            "formatter": "json",
            "stream": "ext://sys.stdout",
        },
        "access": {
            "class": "logging.StreamHandler",
            "formatter": "plain",
            "stream": "ext://sys.stdout",
        },
    },
    "loggers": {
        "gunicorn.error": {
            "handlers": ["stdout"],
            "level": "INFO",
            "propagate": False,
        },
        "gunicorn.access": {
            "handlers": ["access"],
            "level": "INFO",
            "propagate": False,
        },
    },
    "root": {
        "handlers": ["stdout"],
        "level": "INFO",
    },
}
