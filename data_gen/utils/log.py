import logging
import os
from logging.handlers import RotatingFileHandler

def get_logger(name, log_file=None, log_level=logging.INFO, file_mode='a', max_bytes=10*1024*1024, backup_count=5):
    """
    Initializes and returns a logger with the specified configurations.
    Ensures no duplicate log entries are created.
    """
    # Get or create logger
    logger = logging.getLogger(name)
    
    # If logger already has handlers, return it to prevent duplicates
    if logger.handlers:
        return logger
    
    # Set the logger's level
    logger.setLevel(log_level)
    
    # Prevent log messages from being propagated to parent loggers
    logger.propagate = False
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    
    # Create handlers list
    handlers = [console_handler]
    
    # Add file handler if log_file is specified
    if log_file:
        # Ensure log directory exists
        os.makedirs(os.path.dirname(log_file), exist_ok=True) if os.path.dirname(log_file) else None
        
        file_handler = RotatingFileHandler(
            log_file, 
            mode=file_mode, 
            maxBytes=max_bytes, 
            backupCount=backup_count
        )
        file_handler.setLevel(log_level)
        handlers.append(file_handler)
    
    # Create formatter
    formatter = logging.Formatter(
        '[%(asctime)s] %(levelname)s - %(name)s - %(filename)s: %(funcName)s - %(lineno)d: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Add formatter to handlers and add handlers to logger
    for handler in handlers:
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger

if __name__ == "__main__":
    logger = get_logger("my_logger", log_file="logs/app.log", log_level=logging.DEBUG)
    logger.info("This is an info message.")
    logger.debug("This is a debug message.")
    logger.warning("This is a warning message.")
    logger.error("This is an error message.")