import os
import cv2  
import whisper
import logging
import coloredlogs
import json
import numpy as np
from typing import List, Dict, Any

# Configure logging
coloredlogs.install(level='INFO', fmt='%(asctime)s - %(filename)s - %(funcName)s - %(levelname)s - %(message)s')

def transcribe_audio(audio_path: str, transcript_dir: str = "data/transcripts", model_name: str = "base") -> Dict[str, Any] | None:
    """
    Transcribes audio using OpenAI Whisper and returns segments with timestamps.

    Args:
        audio_path: Path to the audio file.
        transcript_dir: Directory to save the transcript.
        model_name: Name of the Whisper model to use (e.g., "tiny", "base", "small", "medium", "large").

    Returns:
        The complete transcription result dictionary (for direct access to segments and word timestamps),
        or None if transcription fails.
    """
    if not os.path.exists(audio_path):
        logging.error(f"Audio file not found: {audio_path}")
        return None

    os.makedirs(transcript_dir, exist_ok=True)
    raw_transcript_path = os.path.join(transcript_dir, "raw_transcript.json")

    try:
        logging.info(f"Loading Whisper model: {model_name}")
        model = whisper.load_model(model_name)
        logging.info(f"Starting transcription for {audio_path}")
        # Set fp16=False if running on CPU or GPU without fp16 support
        result = model.transcribe(audio_path, fp16=False, verbose=True, word_timestamps=True)
        logging.info("Transcription complete.")

        # Save the raw result
        with open(raw_transcript_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        logging.info(f"Raw transcript saved to {raw_transcript_path}")

        return result  

    except Exception as e:
        logging.error(f"Error during transcription: {e}")
        return None

def segment_transcript(segments: List[Dict[str, Any]], max_duration_sec: float = 10.0, transcript_dir: str = "data/transcripts", method: str = "time_based") -> List[Dict[str, Any]]:
    """
    Segments the transcript chunks further if needed, delegating to specific segmentation methods.

    Args:
        segments: List of segment dictionaries from Whisper.
        max_duration_sec: Maximum desired duration for merged segments.
        transcript_dir: Directory to save the segmented transcript.
        method: Segmentation method ('time_based', 'word_based', or 'word_based_sentence').

    Returns:
        A list of potentially merged/adjusted segment dictionaries.
    """
    # Check if segments have word-level timestamps for word-based methods
    has_words = False
    for segment in segments:
        if 'words' in segment and segment['words']:
            has_words = True
            break
    
    if (method == "word_based" or method == "word_based_sentence") and not has_words:
        logging.warning(f"{method} requested but word timestamps not found in segments. Falling back to time-based segmentation.")
        method = "time_based"
    
    if method == "word_based":
        logging.info("Using word-based segmentation with word timestamps")
        return segment_by_words(segments, max_duration_sec, transcript_dir)
    elif method == "word_based_sentence":
        logging.info("Using word-based sentence-preserving segmentation")
        return segment_by_word_sentence(segments, max_duration_sec, transcript_dir)
    elif method == "time_based":
        logging.info("Using time-based segmentation")
        return segment_by_time(segments, max_duration_sec, transcript_dir)
    else:
        # If we reach here, an unsupported method was specified
        logging.warning(f"Method '{method}' not implemented. Falling back to time-based segmentation.")
        return segment_by_time(segments, max_duration_sec, transcript_dir)

def segment_by_time(segments: List[Dict[str, Any]], max_duration_sec: float = 10.0, transcript_dir: str = "data/transcripts") -> List[Dict[str, Any]]:
    """
    Segments the transcript using a time-based approach.
    
    Args:
        segments: List of segment dictionaries from Whisper.
        max_duration_sec: Maximum desired duration for merged segments.
        transcript_dir: Directory to save the segmented transcript.
        
    Returns:
        A list of time-based segment dictionaries.
    """
    logging.info("Creating segments using time-based approach")
    
    # Time-based segmentation logic
    merged_segments = []
    current_segment = None

    for segment in segments:
        if current_segment is None:
            # Start a new merged segment with only relevant fields
            current_segment = {
                'start': segment['start'],
                'end': segment['end'],
                'text': segment['text']
            }
            continue

        # Check if adding the next segment exceeds max duration
        if (segment['end'] - current_segment['start']) <= max_duration_sec:
            # Merge segment into current_segment
            current_segment['text'] += segment['text']
            current_segment['end'] = segment['end']
        else:
            # Finalize the current merged segment
            merged_segments.append(current_segment)
            # Start a new merged segment with only relevant fields
            current_segment = {
                'start': segment['start'],
                'end': segment['end'],
                'text': segment['text']
            }

    # Add the last processed segment
    if current_segment is not None:
        merged_segments.append(current_segment)

    # Add a unique ID to each segment
    for i, seg in enumerate(merged_segments):
        seg['id'] = f"segment_{i}"

    # Save segmented transcript
    segmented_path = os.path.join(transcript_dir, "segmented_transcript.json")
    try:
        with open(segmented_path, 'w', encoding='utf-8') as f:
             json.dump(merged_segments, f, indent=2, ensure_ascii=False)
        logging.info(f"Time-based segmented transcript saved to {segmented_path}")
    except Exception as e:
        logging.error(f"Could not save time-based segmented transcript: {e}")

    logging.info(f"Original segments: {len(segments)}, Time-based segments: {len(merged_segments)}")
    return merged_segments

def segment_by_words(segments: List[Dict[str, Any]], max_duration_sec: float = 10.0, transcript_dir: str = "data/transcripts") -> List[Dict[str, Any]]:
    """
    Creates segments based on word-level timestamps.
    This allows for more precise segment boundaries.
    
    Args:
        segments: List of segment dictionaries from Whisper with word-level timestamps.
        max_duration_sec: Maximum desired duration for segments.
        transcript_dir: Directory to save the segmented transcript.
        
    Returns:
        A list of word-based segment dictionaries.
    """
    logging.info("Creating segments using word-level timestamps")
    
    word_based_segments = []
    current_segment = None
    current_text = []
    
    # Iterate through each segment
    for segment in segments:
        if 'words' not in segment or not segment['words']:
            continue
            
        # Process each word in the segment
        for word in segment['words']:
            if not word.get('word'):  # Skip empty words
                continue
                
            if current_segment is None:
                # Start a new segment
                current_segment = {
                    'start': word['start'],
                    'end': word['end'],
                }
                current_text = [word['word']]
            elif word['end'] - current_segment['start'] <= max_duration_sec:
                # Add word to current segment
                current_text.append(word['word'])
                current_segment['end'] = word['end']
            else:
                # Finalize current segment and start a new one
                current_segment['text'] = ''.join(current_text)
                word_based_segments.append(current_segment)
                
                # Start new segment
                current_segment = {
                    'start': word['start'],
                    'end': word['end'],
                }
                current_text = [word['word']]
    
    # Add the last segment if exists
    if current_segment is not None:
        current_segment['text'] = ''.join(current_text)
        word_based_segments.append(current_segment)
    
    # Add a unique ID to each segment
    for i, seg in enumerate(word_based_segments):
        seg['id'] = f"segment_{i}"
    
    # Save word-based segmented transcript
    word_segmented_path = os.path.join(transcript_dir, "segmented_transcript.json")
    try:
        with open(word_segmented_path, 'w', encoding='utf-8') as f:
             json.dump(word_based_segments, f, indent=2, ensure_ascii=False)
        logging.info(f"Word-based segmented transcript saved to {word_segmented_path}")
    except Exception as e:
        logging.error(f"Could not save word-based segmented transcript: {e}")
    
    logging.info(f"Original segments: {len(segments)}, Word-based segments: {len(word_based_segments)}")
    return word_based_segments

def segment_by_word_sentence(segments: List[Dict[str, Any]], max_duration_sec: float = 10.0, transcript_dir: str = "data/transcripts") -> List[Dict[str, Any]]:
    """
    Creates segments based on word-level timestamps while preserving sentence boundaries.
    Uses a simpler approach:
    1. Add words to the segment until max duration is reached
    2. After reaching max duration, continue adding words until punctuation is found
    3. If a hard limit is reached, break regardless of punctuation
    
    Args:
        segments: List of segment dictionaries from Whisper with word-level timestamps.
        max_duration_sec: Maximum desired duration for segments.
        transcript_dir: Directory to save the segmented transcript.
        
    Returns:
        A list of sentence-aware word-based segment dictionaries.
    """
    logging.info("Creating segments using word-level timestamps with sentence preservation")
    
    # First, collect all words from all segments
    all_words = []
    for segment in segments:
        if 'words' not in segment or not segment['words']:
            continue
        
        for word in segment['words']:
            if not word.get('word'):  # Skip empty words
                continue
            all_words.append(word)
    
    if not all_words:
        logging.warning("No word-level timestamps found in segments")
        return segment_by_time(segments, max_duration_sec, transcript_dir)
    
    logging.info(f"Found {len(all_words)} words with timestamps")
    
    # Create sentence-based segments
    sentence_segments = []
    current_segment = None
    current_text = []
    hard_limit_sec = max_duration_sec + 10  # Hard limit to prevent overly long segments
    
    # Punctuation that ends a sentence
    sentence_end_markers = ['.', '!', '?']
    looking_for_punctuation = False
    
    for word in all_words:
        word_text = word['word']
        ends_sentence = any(word_text.strip().endswith(marker) for marker in sentence_end_markers)
        
        # Start a new segment if this is the first word
        if current_segment is None:
            current_segment = {
                'start': word['start'],
                'end': word['end'],
            }
            current_text = [word_text]
            # If first word already meets duration criteria, start looking for punctuation
            if word['end'] - word['start'] > max_duration_sec:
                looking_for_punctuation = True
        else:
            # Calculate duration with current word
            duration = word['end'] - current_segment['start']
            
            # Case 1: We're looking for punctuation and found it, or hit hard limit
            if (looking_for_punctuation and ends_sentence) or duration > hard_limit_sec:
                # Add the current word to finish the sentence
                current_text.append(word_text)
                current_segment['end'] = word['end']
                current_segment['text'] = ''.join(current_text)
                sentence_segments.append(current_segment)
                
                # Reset for next segment
                current_segment = None
                current_text = []
                looking_for_punctuation = False
            
            # Case 2: We've reached the duration limit and should start looking for punctuation
            elif not looking_for_punctuation and duration > max_duration_sec:
                current_text.append(word_text)
                current_segment['end'] = word['end']
                looking_for_punctuation = True
                
                # If this word happens to end with punctuation, close the segment immediately
                if ends_sentence:
                    current_segment['text'] = ''.join(current_text)
                    sentence_segments.append(current_segment)
                    current_segment = None
                    current_text = []
                    looking_for_punctuation = False
            
            # Case 3: Continue building the current segment
            else:
                current_text.append(word_text)
                current_segment['end'] = word['end']
    
    # Add the last segment if exists
    if current_segment is not None:
        current_segment['text'] = ''.join(current_text)
        sentence_segments.append(current_segment)
    
    # Add a unique ID to each segment
    for i, seg in enumerate(sentence_segments):
        seg['id'] = f"segment_{i}"
    
    # Save sentence-aware segmented transcript
    sentence_segmented_path = os.path.join(transcript_dir, "segmented_transcript.json")
    try:
        with open(sentence_segmented_path, 'w', encoding='utf-8') as f:
             json.dump(sentence_segments, f, indent=2, ensure_ascii=False)
        logging.info(f"Sentence-aware segmented transcript saved to {sentence_segmented_path}")
    except Exception as e:
        logging.error(f"Could not save sentence-aware segmented transcript: {e}")
    
    logging.info(f"Original segments: {len(segments)}, Sentence-aware segments: {len(sentence_segments)}")
    return sentence_segments

def extract_keyframes(video_path: str, keyframe_dir: str, interval_sec: int = 5, detect_scenes: bool = False, threshold: float = 30.0) -> List[Dict[str, Any]]:
    """
    Extracts keyframes from a video at a specified interval or when scene changes are detected.

    Args:
        video_path: Path to the video file.
        keyframe_dir: Directory to save the extracted keyframes.
        interval_sec: Interval in seconds between keyframe extractions (used if detect_scenes is False).
        detect_scenes: If True, detect scene changes instead of using fixed intervals.
        threshold: Threshold for scene change detection (higher = less sensitive).

    Returns:
        A list of dictionaries, each containing 'frame_path', 'timestamp_sec', 'start_time' and 'end_time' for scenes.
    """
    if not os.path.exists(video_path):
        logging.error(f"Video file not found: {video_path}")
        return []

    os.makedirs(keyframe_dir, exist_ok=True)
    keyframes_info = []

    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logging.error(f"Error opening video file: {video_path}")
            return []

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_sec = total_frames / fps if fps > 0 else 0
        
        logging.info(f"Video info: {total_frames} frames, {fps} fps, {duration_sec:.2f} seconds")
        
        frame_interval = int(fps * interval_sec)
        frame_count = 0
        saved_count = 0
        prev_frame = None
        
        # For scene detection with start/end times
        current_scene_start = 0
        scene_frames = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break # End of video
                
            timestamp_sec = frame_count / fps
            
            # Determine if we should save this frame
            save_frame = False
            new_scene_detected = False
            
            if detect_scenes:
                # Scene change detection
                if prev_frame is not None:
                    # Convert to grayscale for faster processing
                    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    gray_prev = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
                    
                    # Calculate mean squared error between frames
                    err = np.sum((gray_frame.astype("float") - gray_prev.astype("float")) ** 2)
                    err /= float(gray_frame.shape[0] * gray_frame.shape[1])
                    
                    # If change is significant enough, mark as a scene change
                    if err > threshold:
                        save_frame = True
                        new_scene_detected = True
                        logging.debug(f"Scene change detected at {timestamp_sec:.2f}s (diff: {err:.2f})")
                        
                        # If we have a previous scene, finalize its end time
                        if scene_frames and len(scene_frames) > 0:
                            scene_end = timestamp_sec  # End time is right before the new scene
                            # Update the previous scene's metadata with its end time
                            keyframes_info[-1]['end_time'] = scene_end
                            
                        # Set the start of the new scene
                        current_scene_start = timestamp_sec
                
                # Always save the first frame as the first scene
                if prev_frame is None:
                    save_frame = True
                    new_scene_detected = True
                    current_scene_start = 0  # Start of the video
            else:
                # Original time-based extraction
                save_frame = frame_count % frame_interval == 0
            
            if save_frame:
                frame_filename = f"frame_{saved_count:05d}_time_{timestamp_sec:.2f}.jpg"
                frame_path = os.path.join(keyframe_dir, frame_filename)

                if cv2.imwrite(frame_path, frame):
                    frame_info = {
                        "frame_path": frame_path,
                        "timestamp_sec": timestamp_sec,
                        "id": f"frame_{saved_count:05d}"
                    }
                    
                    # Add start/end times for scene detection
                    if detect_scenes:
                        frame_info["start_time"] = current_scene_start
                        # We don't know the end time yet (will be updated when next scene is detected)
                        # Set a temporary end time as the video duration
                        frame_info["end_time"] = duration_sec
                        
                        if new_scene_detected:
                            scene_frames.append(frame_count)
                    
                    keyframes_info.append(frame_info)
                    saved_count += 1
                else:
                    logging.warning(f"Could not write frame {frame_count} to {frame_path}")
            
            # Store current frame for next iteration
            prev_frame = frame.copy()
            frame_count += 1

        cap.release()
        logging.info(f"Extracted {saved_count} keyframes from {video_path}")

        # Save keyframe metadata
        keyframes_meta_path = os.path.join(keyframe_dir, "keyframes_metadata.json")
        try:
            with open(keyframes_meta_path, 'w', encoding='utf-8') as f:
                json.dump(keyframes_info, f, indent=2, ensure_ascii=False)
            logging.info(f"Keyframe metadata saved to {keyframes_meta_path}")
        except Exception as e:
            logging.error(f"Could not save keyframe metadata: {e}")


    except Exception as e:
        logging.error(f"Error during keyframe extraction: {e}")
        if 'cap' in locals() and cap.isOpened():
            cap.release()
        return [] # Return empty list on error

    return keyframes_info