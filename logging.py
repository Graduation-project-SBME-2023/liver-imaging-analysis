
import logging

def setup_logger():

    # Create a logger instance
    logger = logging.getLogger()

    # Ceate a stream handler for the logger
    file_handler =logging.StreamHandler()
    # Create a file handler for app.log
    file_handler = logging.FileHandler('logs/app.log')
    # Set the log level for the file handler
    file_handler.setLevel(logging.DEBUG)

    # Create a formatter to specify the log message format
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s] (%(filename)s:%(lineno)d) - %(message)s")

    # Set the formatter for the file handler
    file_handler.setFormatter(formatter)

    # Add the file handler to the logger
    logger.addHandler(file_handler)

    return logger

if __name__ == '__main__':
    
    logger = setup_logger() 
