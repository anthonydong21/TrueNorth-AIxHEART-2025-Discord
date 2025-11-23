import logging
import os
import inspect
import sys


def get_caller_logger(to_stdout: bool = True) -> logging.Logger:
    caller_frame = inspect.stack()[1]
    module = inspect.getmodule(caller_frame[0])
    logger_name = module.__name__ if module else "__main__"
    file_path = os.path.dirname(os.path.realpath(__file__))
    log_dir = os.path.join(os.path.dirname(file_path), ".logs")
    log_file = os.path.join(log_dir, "agent_pipeline.log")

    # Create .logs directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] [%(funcName)s] %(message)s")

    # Prevent adding duplicate handlers
    if not logger.handlers:
        # File handler
        if os.getenv("LOG_TO_FILE", "true").lower() == "true":
            file_handler = logging.FileHandler(log_file, mode="a")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        # Optional stdout handler
        if to_stdout:
            stream_handler = logging.StreamHandler(sys.stdout)
            stream_handler.setFormatter(formatter)
            logger.addHandler(stream_handler)

    return logger
