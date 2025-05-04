import logging
import coloredlogs
from sentence_transformers import SentenceTransformer
from PIL import Image
from typing import List, Dict, Any

coloredlogs.install(level='INFO', fmt='%(asctime)s - %(filename)s - %(funcName)s - %(levelname)s - %(message)s')

def load_text_model(model_name: str) -> SentenceTransformer:
    """Loads a Sentence Transformer model for text embeddings."""
    logging.info(f"Loading text embedding model: {model_name}")
    model = SentenceTransformer(model_name, device="cpu", trust_remote_code=True)
    logging.info(f"Text model {model_name} loaded successfully.")
    return model

def load_vision_model(model_name: str) -> SentenceTransformer:
    """Loads a Sentence Transformer model for image embeddings (e.g., CLIP)."""
    logging.info(f"Loading vision embedding model: {model_name}")
    model = SentenceTransformer(model_name, device="cpu")
    logging.info(f"Vision model {model_name} loaded successfully.")
    return model

def generate_text_embeddings(segments: List[Dict[str, Any]], text_model: SentenceTransformer) -> List[Dict[str, Any]]:
    """
    Generates embeddings for text segments.

    Args:
        segments: List of segment dictionaries (must contain 'text' and 'id').
        text_model: Loaded Sentence Transformer model for text.

    Returns:
        A list of dictionaries, each containing 'id', 'text', 'start', 'end',
        and 'embedding' (as a list or numpy array).
    """
    if not segments:
        logging.warning("No text segments provided for embedding.")
        return []

    logging.info(f"Generating text embeddings for {len(segments)} segments...")
    segment_texts = [seg['text'] for seg in segments]

    embeddings = text_model.encode(segment_texts, convert_to_numpy=True, show_progress_bar=True)
    logging.info("Text embeddings generated.")

    embedded_segments = []
    for i, seg in enumerate(segments):
        seg_copy = seg.copy()
        seg_copy['embedding'] = embeddings[i].tolist()
        embedded_segments.append(seg_copy)

    return embedded_segments

def generate_image_embeddings(keyframes_info: List[Dict[str, Any]], vision_model: SentenceTransformer) -> List[Dict[str, Any]]:
    """
    Generates embeddings for keyframe images.

    Args:
        keyframes_info: List of dictionaries (must contain 'frame_path' and 'id').
        vision_model: Loaded Sentence Transformer model for vision (CLIP).

    Returns:
        A list of dictionaries, each containing 'id', 'frame_path', 'timestamp_sec',
        'start_time', 'end_time' (if present), and 'embedding' (as a list or numpy array).
    """
    if not keyframes_info:
        logging.warning("No keyframes provided for embedding.")
        return []

    logging.info(f"Generating image embeddings for {len(keyframes_info)} keyframes...")
    frame_paths = [info['frame_path'] for info in keyframes_info]

    # Open images before encoding (required by sentence-transformers CLIP)
    try:
        images = [Image.open(filepath) for filepath in frame_paths]
    except Exception as e:
        logging.error(f"Error opening image files for embedding: {e}")
        return []

    # Generate embeddings
    embeddings = vision_model.encode(images, convert_to_numpy=True, show_progress_bar=True)
    logging.info("Image embeddings generated.")

    embedded_keyframes = []
    for i, info in enumerate(keyframes_info):
        info_copy = info.copy()
        info_copy['embedding'] = embeddings[i].tolist()
        
        # Log if we have scene detection data
        if 'start_time' in info and 'end_time' in info:
            logging.debug(f"Keyframe {info.get('id')} has scene data: start={info.get('start_time')}, end={info.get('end_time')}")

        embedded_keyframes.append(info_copy)

    return embedded_keyframes