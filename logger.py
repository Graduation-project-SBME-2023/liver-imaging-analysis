import os
import logging
import datetime


def setup_logger(name):
    """
    Sets up a logger with specified configurations for logging purposes.

    Args:
        name (str): The name of the logger file.

    Returns:
        Logger: The configured logger instance.

    """
    # Create a logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    run_identifier = datetime.datetime.now().strftime("%Y_%m_%d_%H%M%S")

    # Create a file handler to save the log output
    if not os.path.exists("logs/"):
        os.makedirs("logs/")
    file_handler = logging.FileHandler(f'logs/app_{name}__{run_identifier}.log')

    # Create a formatter and add it to the file handler
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s] (%(filename)s:%(lineno)d) - %(message)s")
    file_handler.setFormatter(formatter)

    # Add the file handler to the logger
    logger.addHandler(file_handler)

    # Disable printing from the library
    logging.getLogger("PIL.PngImagePlugin").propagate = False

    # Stop printing from the library
    logging.getLogger("PIL.PngImagePlugin").setLevel(logging.CRITICAL)
    return logger