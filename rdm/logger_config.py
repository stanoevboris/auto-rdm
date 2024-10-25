import sys
import logging
import logging.config
import os
import yaml
from datetime import datetime

PROJECT_DIR = os.path.dirname(__file__)


class StreamToLogger:
    """
    Redirects writes to a logger instance.
    """

    def __init__(self, logger, log_level=logging.INFO):
        self.logger = logger
        self.log_level = log_level

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.log_level, line.rstrip())

    def flush(self):
        pass  # No-op for this implementation


def setup_logging(
        default_path='logging.yaml',
        default_level=logging.INFO,
        env_key='LOG_CFG'
):
    """Setup logging configuration"""
    path = default_path
    value = os.getenv(env_key, None)
    if value:
        path = value
    if os.path.exists(path):
        with open(path, 'rt') as f:
            config = yaml.safe_load(f.read())
        logging.config.dictConfig(config)
    else:
        logger = logging.getLogger()
        if not logger.hasHandlers():  # Ensure handlers are only added once
            logging.basicConfig(level=default_level)
            logger.setLevel(default_level)

            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_filename = f"logs_{timestamp}.log"
            log_dir = os.path.join(PROJECT_DIR, "logs")
            os.makedirs(log_dir, exist_ok=True)

            file_handler = logging.FileHandler(f"{log_dir}/{log_filename}")
            file_handler.setLevel(logging.INFO)
            formatter = logging.Formatter(fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                                          datefmt='%Y-%m-%d %H:%M:%S')
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.DEBUG)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)

    return logging.getLogger()


def handle_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        # Do not log keyboard interrupts
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    logger.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))


# Override the default exception handler
sys.excepthook = handle_exception
# Initialize the logger when this module is imported
logger = setup_logging()
# Redirect stdout and stderr to the logger
sys.stdout = StreamToLogger(logger, logging.INFO)
sys.stderr = StreamToLogger(logger, logging.ERROR)

