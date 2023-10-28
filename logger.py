import os
import logging
import datetime


class ExcludeLogFilter(logging.Filter):
    def filter(self, record):
        # Exclude log messages containing 'STREAM b'
        if "STREAM b" in record.getMessage():
            return False
        # Exclude log messages from logger 'PIL.PngImagePlugin'
        if record.name.startswith('PIL.PngImagePlugin'):
            return False
        return True

def setup_logger():
    # Create a logger instance for the app
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # Generate a timestamp or run identifier
    run_identifier = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

    # Create a file handler for app.log with the encoded run identifier
    if not os.path.exists("logs/"):
        os.makedirs("logs/")
    file_handler = logging.FileHandler(f'logs/app_{run_identifier}.log')
    file_handler.setLevel(logging.DEBUG)


    # Create an instance of the custom filter
    exclude_filter = ExcludeLogFilter()

    # Add the filter to the file handler
    file_handler.addFilter(exclude_filter)


    # Create a formatter to specify the log message format
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s] (%(filename)s:%(lineno)d) - %(message)s")

    # Set the formatter for the file handler
    file_handler.setFormatter(formatter)

    # Add the file handler to the logger
    logger.addHandler(file_handler)

    # Add the custom filter to exclude specific log messages
    logger.addFilter(ExcludeLogFilter())

setup_logger()