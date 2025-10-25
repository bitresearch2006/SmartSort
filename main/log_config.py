import os
import sys
import logging

# Initialize logger globally once at the module level
logger = logging.getLogger(__name__)

def initialize_logger(log_name="FaaS", default_log_file="FaaS.log"):
    """
    Initializes the logger based on the DIAGNOSTICS environment variable.

    When DIAGNOSTICS=true:
    - Logs are written to /tmp/{LOG_FILE}, where LOG_FILE defaults to 'SmartSort.log'.
    - Critical logs are also displayed in the console.

    Args:
        log_name (str): The name for the logger instance.
        default_log_file (str): The default filename if LOG_FILE is not set.
    """
    global logger
    logger = logging.getLogger(log_name)
    logger.setLevel(logging.INFO)
    
    # Ensure handlers are cleared to prevent duplicates if called multiple times
    if logger.handlers:
        return logger

    diagnostics_mode = os.getenv("DIAGNOSTICS", "false").lower() == "true"
    
    if diagnostics_mode:
        logger.propagate = False 

        # Simple formatter for console/output logging
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

        # Define log file path using environment variable LOG_FILE
        log_dir = "/tmp"
        # Use LOG_FILE environment variable, falling back to the default provided argument
        log_file = os.getenv("LOG_FILE", default_log_file) 
        log_path = os.path.join(log_dir, log_file)
        
        os.makedirs(log_dir, exist_ok=True)

        # 1. File Handler
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        # 2. Console Handler (for critical issues)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(logging.WARNING)
        logger.addHandler(console_handler)
        
        logger.info(f"Diagnostics mode enabled. Logs are being written to {log_path}")
    else:
        # If not in diagnostics mode, disable all but critical logging
        logger.setLevel(logging.CRITICAL)
        
    return logger

# Initialize the logger immediately when the module is imported
logger = initialize_logger()
