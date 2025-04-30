# /training/graph/utils.py

import torch
import random
import numpy as np
import os
import json
import logging

def set_seed(seed: int):
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Potentially make cuDNN deterministic (can slow down training)
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False

def setup_logging(log_dir: str, log_filename: str = "training.log"):
    """Configures logging to file and console."""
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_filename)

    # Find existing handlers to avoid adding duplicates if called multiple times
    # (Optional but good practice if the script might call setup_logging multiple times)
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
        handler.close()

    logging.basicConfig(
        level=logging.DEBUG, # <--- GARANTE QUE ESTÁ DEBUG
        format="%(asctime)s [%(levelname)s] (%(module)s:%(lineno)d) %(message)s", # Formato útil
        handlers=[
            logging.FileHandler(log_path, mode='w'), # Sobrescreve o log a cada execução
            logging.StreamHandler() # Log para consola
        ]
    )
    logging.info("Logging setup complete (Level: DEBUG).") # Mensagem atualizada
    logging.info(f"Logs will be saved to: {log_path}")

def save_config(args, output_dir: str):
    """Saves the training configuration (argparse args) to a JSON file."""
    config_path = os.path.join(output_dir, 'config.json')
    # Convert Namespace to dict, handle non-serializable items like device
    args_dict = vars(args).copy()
    if 'device' in args_dict:
        args_dict['device'] = str(args_dict['device']) # Convert device object to string

    try:
        with open(config_path, 'w') as f:
            json.dump(args_dict, f, indent=4)
        logging.info(f"Training configuration saved to {config_path}")
    except Exception as e:
        logging.error(f"Failed to save configuration: {e}")

def load_config(config_path: str):
    """Loads a training configuration from a JSON file."""
    if not os.path.exists(config_path):
         logging.error(f"Configuration file not found: {config_path}")
         return None
    try:
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        # You might want to convert back to argparse Namespace if needed,
        # or just use the dictionary directly.
        logging.info(f"Loaded configuration from {config_path}")
        return config_dict
    except Exception as e:
         logging.error(f"Failed to load configuration: {e}")
         return None

# Add more utility functions as needed (e.g., plotting loss curves)