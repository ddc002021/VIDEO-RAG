########################################################################
################################## SETUP ###############################
########################################################################

import logging
import coloredlogs
import os
import sys

coloredlogs.install(level='INFO', fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

try:
    from utils.constants import CONFIG_PATH
    from core.embedding import load_text_model, load_vision_model, generate_text_embeddings, generate_image_embeddings
    from utils.utils import load_config, load_json_data, save_json_data
except ImportError as e:
    logging.error(f"Error importing modules: {e}")
    sys.exit(1)


########################################################################
############################# MAIN BLOCK ###############################
########################################################################

def main():
    logging.info("=== Starting Embedding Generation Pipeline ===")
    config = load_config(CONFIG_PATH)
    if config is None:
        logging.critical("Failed to load configuration. Exiting.")
        return

    # --- Define Input/Output Paths ---
    data_dir = config.get('data_dir', 'data')
    transcript_dir = config.get('transcript_dir', os.path.join(data_dir, 'transcripts'))
    keyframe_dir = config.get('keyframe_dir', os.path.join(data_dir, 'keyframes'))
    embedding_dir = os.path.join(data_dir, 'embeddings')

    segmented_transcript_path = os.path.join(transcript_dir, "segmented_transcript.json")
    keyframes_metadata_path = os.path.join(keyframe_dir, "keyframes_metadata.json")

    text_embedding_output_path = os.path.join(embedding_dir, "text_embeddings.json")
    image_embedding_output_path = os.path.join(embedding_dir, "image_embeddings.json")

    # --- Load Processed Data ---
    logging.info("=== Loading Processed Data ===")
    segments = load_json_data(segmented_transcript_path)
    keyframes_info = load_json_data(keyframes_metadata_path)

    # Check if data loading was successful and data is not empty
    if segments is None or not isinstance(segments, list):
        logging.error(f"Failed to load or validate segmented transcripts from {segmented_transcript_path}. Exiting.")
        return
    if keyframes_info is None or not isinstance(keyframes_info, list):
        logging.error(f"Failed to load or validate keyframes metadata from {keyframes_metadata_path}. Exiting.")
        return

    if not segments:
        logging.warning("Segmented transcript file is empty. No text embeddings will be generated.")
    if not keyframes_info:
        logging.warning("Keyframes metadata file is empty. No image embeddings will be generated.")

    # --- Load Embedding Models ---
    logging.info("=== Loading Embedding Models ===")
    text_model_name = config.get('text_embedding_model')
    vision_model_name = config.get('vision_embedding_model')

    if not text_model_name:
        logging.error("Config Error: 'text_embedding_model' not specified in config.yaml. Cannot generate text embeddings.")
        return
    else:
        try:
            text_model = load_text_model(text_model_name)
            if text_model is None: raise ValueError("Text model loading returned None")
        except Exception as e:
            logging.error(f"Failed to load text embedding model '{text_model_name}': {e}", exc_info=True)
            return

    if not vision_model_name:
        logging.error("Config Error: 'vision_embedding_model' not specified in config.yaml. Cannot generate image embeddings.")
        return
    else:
        try:
            vision_model = load_vision_model(vision_model_name)
            if vision_model is None: raise ValueError("Vision model loading returned None")
        except Exception as e:
            logging.error(f"Failed to load vision embedding model '{vision_model_name}': {e}", exc_info=True)
            return

    # --- Generate Text Embeddings ---
    if segments:
        logging.info("=== Generating Text Embeddings ===")
        try:
            required_keys = ['id', 'text']
            valid_segments = [s for s in segments if all(k in s for k in required_keys)]
            if len(valid_segments) < len(segments):
                 logging.warning(f"{len(segments) - len(valid_segments)} segments missing required keys ('id', 'text').")

            if valid_segments:
                 text_embeddings_data = generate_text_embeddings(valid_segments, text_model)
                 if text_embeddings_data:
                     logging.info(f"Successfully generated {len(text_embeddings_data)} text embeddings.")
                     save_json_data(text_embeddings_data, text_embedding_output_path)
                 else:
                     logging.warning("Text embedding generation produced no output data.")
            else:
                 logging.warning("No valid segments found for text embedding generation.")

        except Exception as e:
            logging.error(f"An error occurred during text embedding generation: {e}", exc_info=True)
            return
    else:
        logging.info("Skipping text embedding generation as no segments were loaded.")


    # --- Generate Image Embeddings ---
    if vision_model:
        logging.info("=== Generating Image Embeddings ===")
        try:
            required_keys = ['id', 'frame_path']
            valid_keyframes = [k for k in keyframes_info if all(key in k for key in required_keys)]
            if len(valid_keyframes) < len(keyframes_info):
                 logging.warning(f"{len(keyframes_info) - len(valid_keyframes)} keyframes missing required keys ('id', 'frame_path').")

            if valid_keyframes:
                image_embeddings_data = generate_image_embeddings(valid_keyframes, vision_model)
                if image_embeddings_data:
                    logging.info(f"Successfully generated {len(image_embeddings_data)} image embeddings.")
                    save_json_data(image_embeddings_data, image_embedding_output_path)
                else:
                    logging.warning("Image embedding generation produced no output data.")
            else:
                 logging.warning("No valid keyframes found for image embedding generation.")

        except Exception as e:
            logging.error(f"An error occurred during image embedding generation: {e}", exc_info=True)
            return
    else:
        logging.info("Skipping image embedding generation as no keyframe info was loaded.")

    logging.info("=== Embedding Generation Pipeline Finished ===")

if __name__ == "__main__":
    main()