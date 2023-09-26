
import logging

from liver_imaging_analysis.engine.config import Config
from liver_imaging_analysis.engine.engine import Engine
from liver_imaging_analysis.models.liver_segmentation import LiverSegmentation
from test_engine import TestEngine

def setup_logger():
    # Create a logger instance for the app
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # Create a file handler for app.log
    file_handler = logging.FileHandler('logs/app.log')
    file_handler.setLevel(logging.DEBUG)

    # Create a formatter to specify the log message format
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s] (%(filename)s:%(lineno)d) - %(message)s")
    

    # Set the formatter for the file handler
    file_handler.setFormatter(formatter)

    # Add the file handler to the logger
    logger.addHandler(file_handler)

    return logger

# def engine():
#     engine = Engine()
#     engine.load_data()
#     return engine


def main():
    logger = setup_logger()

    logger.info('LIVER_IMAGIN_ANALYSIS')

    # config= Config()
    # engine()
    liver_model = TestEngine()


    logger.info('App finished')

if __name__ == '__main__':
    main()