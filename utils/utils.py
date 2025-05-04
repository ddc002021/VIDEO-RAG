import yaml
import logging
import coloredlogs
import os
import json
import numpy as np
from typing import List, Optional
coloredlogs.install(level='INFO', fmt='%(asctime)s - %(filename)s - %(funcName)s - %(levelname)s - %(message)s')

def load_config(config_path):
    """Loads configuration from a YAML file."""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        logging.info(f"Configuration loaded from {config_path}")
        return config
    except FileNotFoundError:
        logging.error(f"ERROR: Configuration file not found at {config_path}")
        return None
    except yaml.YAMLError as e:
        logging.error(f"ERROR: Error parsing configuration file {config_path}: {e}")
        return None
    except Exception as e:
        logging.error(f"ERROR: An unexpected error occurred loading configuration: {e}")
        return None
    
def load_json_data(file_path):
    """Loads data from a JSON file."""
    if not os.path.exists(file_path):
         logging.error(f"ERROR: File not found at {file_path}")
         return None
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logging.info(f"Successfully loaded data from {file_path}")
        return data
    except json.JSONDecodeError:
        logging.error(f"ERROR: Could not decode JSON from {file_path}")
        return None
    except Exception as e:
        logging.error(f"ERROR: An unexpected error occurred loading {file_path}: {e}")
        return None

def save_json_data(data, file_path):
    """Saves data to a JSON file, creating directories if needed."""
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            # Use ensure_ascii=False for broader character support, indent for readability
            json.dump(data, f, indent=2, ensure_ascii=False)
        logging.info(f"Successfully saved data to {file_path}")
        return True
    except TypeError as e:
         logging.error(f"ERROR: Data is not JSON serializable. Check for numpy arrays or other non-standard types: {e}")
         return False
    except Exception as e:
        logging.error(f"ERROR: Could not save data to {file_path}: {e}")
        return False
    
def load_embeddings_as_numpy(filepath):
    """Loads embeddings from JSON and converts 'embedding' list to numpy array."""
    data = load_json_data(filepath)
    if data is None:
        return None, 0 # Return None and dimension 0 if loading failed

    converted_data = []
    embedding_dim = 0
    first_embedding_found = False

    for item in data:
        if 'embedding' in item and isinstance(item['embedding'], list):
            try:
                embedding_array = np.array(item['embedding']).astype('float32') # FAISS prefers float32
                item['embedding'] = embedding_array
                converted_data.append(item)
                if not first_embedding_found:
                    embedding_dim = embedding_array.shape[0]
                    first_embedding_found = True
                elif embedding_array.shape[0] != embedding_dim:
                     logging.warning(f"Inconsistent embedding dimension found in {filepath}. Expected {embedding_dim}, got {embedding_array.shape[0]}. Skipping item ID: {item.get('id', 'N/A')}")
                     converted_data.pop() # Remove the problematic item
            except Exception as e:
                logging.error(f"Error converting embedding to numpy for item ID {item.get('id', 'N/A')} in {filepath}: {e}")
        else:
             logging.warning(f"Item ID {item.get('id', 'N/A')} in {filepath} missing 'embedding' list or invalid format. Skipping.")

    if not first_embedding_found and data: # Check if data was loaded but no valid embeddings found
        logging.error(f"No valid embeddings found in {filepath} to determine dimension.")
        return None, 0

    logging.info(f"Embeddings loaded and converted from {filepath}. Dimension: {embedding_dim}")
    return converted_data, embedding_dim

def get_embedding_dimensions(config):
    """Determine embedding dimensions"""
    text_dim = None
    image_dim = None
    
    data_dir = config.get('data_dir', 'data')

    text_embeddings_path = os.path.join(data_dir, 'embeddings', 'text_embeddings.json')
    if os.path.exists(text_embeddings_path):
        try:
            _, extracted_text_dim = load_embeddings_as_numpy(text_embeddings_path)
            text_dim = extracted_text_dim
        except Exception as e:
            logging.warning(f"Could not extract text embedding dimension from file: {e}")
    
    image_embeddings_path = os.path.join(data_dir, 'embeddings', 'image_embeddings.json')
    if os.path.exists(image_embeddings_path):
        try:
            _, extracted_image_dim = load_embeddings_as_numpy(image_embeddings_path)
            image_dim = extracted_image_dim
        except Exception as e:
            logging.warning(f"Could not extract image embedding dimension from file: {e}")
    
    return text_dim, image_dim

def parse_timestamp(ts_str: str) -> float | None:
    """Converts HH:MM:SS or seconds string/number to float seconds."""
    if isinstance(ts_str, (int, float)):
        return float(ts_str)
    if isinstance(ts_str, str):
        parts = ts_str.split(':')
        try:
            if len(parts) == 3:
                h, m, s = map(float, parts)
                return h * 3600 + m * 60 + s
            elif len(parts) == 2:
                m, s = map(float, parts)
                return m * 60 + s
            elif len(parts) == 1:
                return float(parts[0])
        except ValueError:
            logging.warning(f"Could not parse timestamp string: {ts_str}")
            return None
    logging.warning(f"Invalid timestamp format: {ts_str}")
    return None

def check_match_text(retrieved_ts_start: Optional[float], retrieved_ts_end: Optional[float], ground_truth_ts_list: List[float]) -> bool:
    """Checks if retrieved timestamp is within tolerance of any ground truth timestamp."""
    if retrieved_ts_start is None or retrieved_ts_end is None or not ground_truth_ts_list:
        return False
    for gt_ts in ground_truth_ts_list:
        if retrieved_ts_start <= gt_ts <= retrieved_ts_end:
            return True
    return False

def check_match_image(retrieved_ts: float, ground_truth_secs: List[float], timestamp_tolerance: float = 5.0,
                     ts_start: float = None, ts_end: float = None) -> bool:
    """Checks if retrieved timestamp is within tolerance of any ground truth timestamp."""
    if not ground_truth_secs:
        return False
        
    # If start and end times are provided, use text-like temporal overlap matching
    if ts_start is not None and ts_end is not None:
        return check_match_text(ts_start, ts_end, ground_truth_secs)
    
    # Otherwise use tolerance-based matching
    if retrieved_ts is None:
        return False
        
    for gt_sec in ground_truth_secs:
        if abs(retrieved_ts - gt_sec) <= timestamp_tolerance:
            return True
    
    return False