import logging
import os
from datetime import datetime

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

def get_logger(name: str):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Generate the dynamic filename based on today's date
    today_date = datetime.now().strftime("%Y-%m-%d")
    log_file_path = os.path.join(LOG_DIR, f"{today_date}_logs.txt")

    # Check if a FileHandler for today's file already exists to prevent duplicate logs
    has_current_file_handler = any(
        isinstance(h, logging.FileHandler) and h.baseFilename == os.path.abspath(log_file_path)
        for h in logger.handlers
    )

    # If the handlers aren't set up yet, or the date has changed, configure them
    if not logger.handlers or not has_current_file_handler:
        # Clear existing handlers if the date changed to avoid duplicating console/file outputs
        logger.handlers.clear()

        # Create handlers
        file_handler = logging.FileHandler(log_file_path)
        console_handler = logging.StreamHandler()

        # Define format
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # Add handlers to logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger