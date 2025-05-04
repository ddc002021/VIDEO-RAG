########################################################################
################################## SETUP ###############################
########################################################################

import logging
import coloredlogs
import os
import sys

coloredlogs.install(level='INFO', fmt='%(asctime)s - %(filename)s - %(funcName)s - %(levelname)s - %(message)s')

try:
    from utils.constants import CONFIG_PATH, DEFAULT_STT_MODEL_NAME, DEFAULT_SEGMENTATION_METHOD, DEFAULT_MAX_DURATION_SEC, DEFAULT_KEYFRAME_INTERVAL_SEC
    from core.processing import transcribe_audio, segment_transcript, extract_keyframes
    from utils.utils import load_config, load_json_data
except ImportError as e:
    logging.error(f"Error importing modules: {e}")
    sys.exit(1)

########################################################################
############################# MAIN BLOCK ###############################
########################################################################

def main():
    logging.info("=== STARTING DATA PREPROCESSING PIPELINE ===")
    config = load_config(CONFIG_PATH)
    if config is None:
        logging.critical("Failed to load configuration. Exiting.")
        return 

    # --- Step 1: Load Video ---
    logging.info("=== Running Step 1: Loading Video ===")

    video_dir = config.get('video_dir', 'data/videos')
    video_filename = config.get('video_filename', 'sample_video.mp4')
    downloaded_video_path = os.path.join(video_dir, video_filename)
    if not os.path.exists(downloaded_video_path):
        logging.error(f"Video file not found at {downloaded_video_path}. Please check the path.")
        return
    logging.info(f"Video file found at {downloaded_video_path}. Proceeding to extract audio.")

    # --- Step 2: Load Audio ---
    logging.info("=== Running Step 2: Extract Audio ===")
    audio_dir = config.get('audio_dir', 'data/audio')
    audio_filename = config.get('audio_filename', 'extracted_audio.mp3')
    extracted_audio_path = os.path.join(audio_dir, audio_filename)
    if not os.path.exists(audio_dir):
        logging.error(f"Audio file not found at {extracted_audio_path}. Please check the path.")
        return
    logging.info(f"Audio file found at {extracted_audio_path}. Proceeding to transcribe audio.")

    # --- Step 3: Transcribe Audio ---
    logging.info("=== Running Step 3: Transcribe Audio ===")
    stt_model = config.get('stt_model', DEFAULT_STT_MODEL_NAME)
    transcript_dir = config.get('transcript_dir', 'data/transcripts')
    raw_transcript_path = os.path.join(transcript_dir, "raw_transcript.json")

    transcript_result = None
    if os.path.exists(raw_transcript_path):
        logging.info(f"Raw transcript already exists in {raw_transcript_path}. Loading for segmentation.")
        transcript_result = load_json_data(raw_transcript_path)
    else:
        transcript_result = transcribe_audio(extracted_audio_path, transcript_dir, model_name=stt_model)
        if transcript_result:
            logging.info(f"SUCCESS: Transcription complete. Found {len(transcript_result['segments'])} raw segments.")
            logging.info(f"(Raw transcript saved to {raw_transcript_path})")
        else:
            logging.error("FAILURE: Audio transcription failed. Stopping preprocessing.")
            return

    # --- Step 4: Segment Transcript ---
    logging.info("=== Running Step 4: Segment Transcript ===")
    segment_duration = config.get('segment_duration_sec', DEFAULT_MAX_DURATION_SEC)
    segment_method = config.get('segmentation_method', DEFAULT_SEGMENTATION_METHOD) 
    
    # Use the segments from the transcription result
    raw_segments = transcript_result['segments']
    processed_segments = segment_transcript(
        raw_segments, 
        max_duration_sec=segment_duration, 
        method=segment_method, 
        transcript_dir=transcript_dir
    )

    if processed_segments:
        segmented_path = os.path.join(transcript_dir, "segmented_transcript.json")
        logging.info(f"SUCCESS: Segmentation complete ({segment_method}). Resulted in {len(processed_segments)} final segments.")
        logging.info(f"(Segmented transcript saved to {segmented_path})")
    else:
        logging.warning("WARNING: Transcript segmentation did not produce refined segments (or failed). Check the logs for core/processing.py.")
        return

    # --- Step 5: Extract Keyframes ---
    logging.info("=== Running Step 5: Extract Keyframes ===")
    keyframe_interval = config.get('keyframe_interval_sec', DEFAULT_KEYFRAME_INTERVAL_SEC)
    keyframe_dir = config.get('keyframe_dir', 'data/keyframes')
    
    use_scene_detection = config.get('use_scene_detection', True)
    scene_change_threshold = config.get('scene_change_threshold', 30.0)
    
    logging.info(f"Using {'scene detection' if use_scene_detection else 'fixed interval'} for keyframe extraction")
    
    keyframes_info = extract_keyframes(downloaded_video_path, keyframe_dir, interval_sec=keyframe_interval, detect_scenes=use_scene_detection, threshold=scene_change_threshold)

    if keyframes_info:
        logging.info(f"SUCCESS: Keyframe extraction complete. Found {len(keyframes_info)} keyframes.")
        keyframes_meta_path = os.path.join(keyframe_dir, "keyframes_metadata.json")
        logging.info(f"(Keyframes and metadata should be saved in {keyframe_dir}, check {keyframes_meta_path})")
    else:
        logging.error("FAILURE: Keyframe extraction failed.")
        return
    
    logging.info("=== Data Preprocessing Pipeline Finished ===")

if __name__ == "__main__":
    main()