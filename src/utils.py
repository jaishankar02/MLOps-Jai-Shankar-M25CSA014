"""
Utility functions for logging and configuration.
"""
import os
import json
import logging
from datetime import datetime

def setup_logging(log_dir: str = "./logs"):
    """
    Setup logging configuration.
    """
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"training_{timestamp}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)

def save_training_config(config: dict, filepath: str):
    """
    Save training configuration to JSON file.
    """
    with open(filepath, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Configuration saved to: {filepath}")

def load_config(filepath: str) -> dict:
    """
    Load configuration from JSON file.
    """
    with open(filepath, "r") as f:
        return json.load(f)