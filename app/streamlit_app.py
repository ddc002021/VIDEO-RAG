import streamlit as st
import os
import sys
import time
import textwrap
from datetime import datetime
import base64

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from utils.utils import load_config, check_match_text, check_match_image
from utils.constants import CONFIG_PATH
from core.embedding import load_text_model, load_vision_model
from core.retrieval import FAISSRetriever, TFIDFRetriever, BM25Retriever, PGVectorRetriever

# Page configuration
st.set_page_config(
    page_title="Video Content Retriever",
    page_icon="🎬",
    layout="wide"
)

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

if "initialized" not in st.session_state:
    st.session_state.initialized = False
    st.session_state.retrievers = {}
    st.session_state.text_model = None
    st.session_state.vision_model = None

def format_time(seconds):
    """Format seconds into a readable time format (MM:SS)"""
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)
    return f"{minutes:02d}:{seconds:02d}"

def get_video_html(video_path, start_time=0):
    """Generate HTML for embedded video starting at a specific time"""
    video_file = open(video_path, 'rb')
    video_bytes = video_file.read()
    video_file.close()
    
    video_b64 = base64.b64encode(video_bytes).decode()
    
    # Custom HTML with autoplay and starting time
    html = f"""
    <video width="100%" controls autoplay>
        <source src="data:video/mp4;base64,{video_b64}" type="video/mp4">
        Your browser does not support the video tag.
    </video>
    <script>
        // Set current time - executed after video loads
        document.querySelector('video').addEventListener('loadedmetadata', function() {{
            this.currentTime = {start_time};
        }});
    </script>
    """
    return html

@st.cache_resource
def load_models_and_retrievers():
    """Load all models and retrievers (cached to prevent reloading)"""
    st.write("Loading models and retrievers...")
    config = load_config(CONFIG_PATH)
    
    # Load models
    text_model_name = config.get('text_embedding_model')
    vision_model_name = config.get('vision_embedding_model')
    
    text_model = load_text_model(text_model_name)
    vision_model = load_vision_model(vision_model_name)
    
    # Determine embedding dimensions
    from utils.utils import get_embedding_dimensions
    text_emb_dim, image_emb_dim = get_embedding_dimensions(config)
    
    # Load retrievers
    retrievers = {}
    index_dir = config.get('index_dir', 'indexes')
    
    # Load FAISS Text
    try:
        faiss_text_index_path = os.path.join(index_dir, "faiss", "text_embeddings.index")
        faiss_text_retriever = FAISSRetriever(embedding_dim=text_emb_dim, index_path=faiss_text_index_path)
        if faiss_text_retriever.load_index():
            retrievers['FAISS_Text'] = faiss_text_retriever
    except Exception as e:
        st.error(f"Error loading FAISS Text retriever: {e}")
    
    # Load FAISS Image
    try:
        faiss_image_index_path = os.path.join(index_dir, "faiss", "image_embeddings.index")
        faiss_image_retriever = FAISSRetriever(embedding_dim=image_emb_dim, index_path=faiss_image_index_path)
        if faiss_image_retriever.load_index():
            retrievers['FAISS_Image'] = faiss_image_retriever
    except Exception as e:
        st.error(f"Error loading FAISS Image retriever: {e}")
    
    # Load TF-IDF
    try:
        tfidf_index_prefix = os.path.join(index_dir, "lexical", "tfidf")
        tfidf_retriever = TFIDFRetriever(index_path_prefix=tfidf_index_prefix)
        if tfidf_retriever.load_index():
            retrievers['TF-IDF'] = tfidf_retriever
    except Exception as e:
        st.error(f"Error loading TF-IDF retriever: {e}")
    
    # Load BM25
    try:
        bm25_index_path = os.path.join(index_dir, "lexical", "bm25_index.pkl")
        bm25_retriever = BM25Retriever(index_path=bm25_index_path)
        if bm25_retriever.load_index():
            retrievers['BM25'] = bm25_retriever
    except Exception as e:
        st.error(f"Error loading BM25 retriever: {e}")
    
    return config, text_model, vision_model, retrievers

def search_with_retrievers(query, top_k=3):
    """Search across all retrievers and merge results"""
    config = st.session_state.config
    text_model = st.session_state.text_model
    vision_model = st.session_state.vision_model
    retrievers = st.session_state.retrievers
    
    # Get relevance thresholds from config
    relevance_thresholds = config.get('relevance_thresholds', {})
    
    all_results = []
    used_retrievers = []
    
    with st.spinner("Searching for answers..."):
        # Text-based retrievers (generally more accurate for specific questions)
        for retriever_name in ['FAISS_Text', 'BM25', 'TF-IDF']:
            if retriever_name in retrievers:
                used_retrievers.append(retriever_name)
                retriever = retrievers[retriever_name]
                
                # Generate embeddings or search directly
                if retriever_name == 'FAISS_Text':
                    query_embedding = text_model.encode([query], convert_to_numpy=True)
                    results = retriever.search(query_embedding, top_k=top_k*2)  # Get more results for filtering
                else:
                    results = retriever.search(query, top_k=top_k*2)
                
                # Apply relevance threshold filtering
                threshold = relevance_thresholds.get(retriever_name)
                if threshold is not None:
                    # For text-based retrievers, higher score is better
                    results = [(meta, score) for meta, score in results if score >= threshold]
                
                # Add source information
                for meta, score in results:
                    meta['retriever'] = retriever_name
                    meta['score'] = score
                    all_results.append(meta)
        
        # Image-based retrievers (can help with visual concepts)
        for retriever_name in ['FAISS_Image']:
            if retriever_name in retrievers:
                used_retrievers.append(retriever_name)
                retriever = retrievers[retriever_name]
                
                # Generate embeddings
                query_embedding = vision_model.encode([query], convert_to_numpy=True)
                results = retriever.search(query_embedding, top_k=top_k)
                
                # Apply relevance threshold filtering
                threshold = relevance_thresholds.get(retriever_name)
                if threshold is not None:
                    # For FAISS Image, higher score is better
                    results = [(meta, score) for meta, score in results if score >= threshold]
                
                # Add source information
                for meta, score in results:
                    meta['retriever'] = retriever_name
                    meta['score'] = score
                    all_results.append(meta)
    
    # Sort all results by score (descending)
    all_results.sort(key=lambda x: x.get('score', 0), reverse=True)
    
    # Remove duplicates based on timestamp overlap
    unique_results = []
    used_timestamps = set()
    
    for result in all_results:
        # Get time information
        start_time = result.get('start_time') or result.get('start')
        end_time = result.get('end_time') or result.get('end')
        timestamp = result.get('timestamp_sec')
        
        # Skip if no timing information
        if start_time is None and timestamp is None:
            continue
        
        # Check if this timestamp overlaps with existing results
        is_duplicate = False
        timestamp_key = None
        
        if start_time is not None and end_time is not None:
            # For segments with start/end times
            timestamp_key = f"{start_time:.1f}-{end_time:.1f}"
            
            # Check for overlap with existing segments
            for used_ts in used_timestamps:
                if "-" in used_ts:
                    used_start, used_end = map(float, used_ts.split("-"))
                    # Check for overlap
                    if (start_time <= used_end and end_time >= used_start):
                        is_duplicate = True
                        break
        
        elif timestamp is not None:
            # For single keyframes
            timestamp_key = f"{timestamp:.1f}"
            
            # Check for exact timestamp match
            if timestamp_key in used_timestamps:
                is_duplicate = True
        
        # Add to results if not a duplicate
        if not is_duplicate and timestamp_key:
            used_timestamps.add(timestamp_key)
            unique_results.append(result)
            
            # Limit to top_k unique results
            if len(unique_results) >= top_k:
                break
    
    return unique_results, used_retrievers

def display_results(results, query):
    """Display search results with video segments"""
    config = st.session_state.config
    video_dir = config.get('video_dir', 'data/video')
    video_filename = config.get('video_filename', 'complexity_talk.mp4')
    video_path = os.path.join(video_dir, video_filename)
    
    if not os.path.exists(video_path):
        st.error(f"Video file not found at {video_path}. Please check the path.")
        return
    
    if not results:
        st.info("No relevant information found in the video for this question.")
        return
    
    st.success(f"Found {len(results)} relevant segment(s) in the video:")
    
    # Create columns for results
    cols = st.columns(min(len(results), 3))
    
    for i, result in enumerate(results):
        with cols[i % len(cols)]:
            # Get timing information
            start_time = result.get('start_time') or result.get('start')
            end_time = result.get('end_time') or result.get('end')
            timestamp = result.get('timestamp_sec')
            
            # Determine what kind of result this is
            is_segment = start_time is not None and end_time is not None
            is_keyframe = timestamp is not None
            retriever_type = result.get('retriever', 'Unknown')
            
            # Create a card for the result
            with st.container(border=True):
                # Header with timing info
                if is_segment:
                    st.subheader(f"Segment {i+1}: {format_time(start_time)} - {format_time(end_time)}")
                    play_time = start_time
                elif is_keyframe:
                    st.subheader(f"Keyframe {i+1}: {format_time(timestamp)}")
                    play_time = timestamp
                
                # Show text content if available
                if 'text' in result and result['text']:
                    text = result['text']
                    # Truncate if too long
                    if len(text) > 300:
                        text = textwrap.shorten(text, width=300, placeholder="...")
                    st.write(text)
                
                # Show image if it's a keyframe result
                if 'frame_path' in result and result['frame_path']:
                    if os.path.exists(result['frame_path']):
                        st.image(result['frame_path'], caption=f"Frame at {format_time(timestamp)}")
                
                # Display video segment
                st.video(video_path, start_time=play_time)
                
                # Show metadata
                st.caption(f"Found by: {retriever_type} (Score: {result.get('score', 'N/A'):.3f})")

def main():
    # Header
    st.title("🎬 Video Content Retriever")
    st.markdown("Ask questions about the video content and get relevant segments as answers.")
    
    # Sidebar
    with st.sidebar:
        st.header("About")
        st.info(
            "This application allows you to ask natural language questions about "
            "the video content and get relevant segments as answers. It uses "
            "multiple retrieval methods including semantic and lexical search."
        )
        
        st.header("Settings")
        top_k = st.slider("Number of results to show", 1, 5, 3)
        
        # Initialize systems on first run
        if not st.session_state.initialized:
            with st.spinner("Loading models and systems..."):
                config, text_model, vision_model, retrievers = load_models_and_retrievers()
                
                st.session_state.config = config
                st.session_state.text_model = text_model
                st.session_state.vision_model = vision_model
                st.session_state.retrievers = retrievers
                st.session_state.initialized = True
                
                st.success(f"Loaded {len(retrievers)} retrievers successfully!")
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                st.markdown(message["content"])
            else:
                # For assistant messages with results
                if "results" in message:
                    display_results(message["results"], message["content"])
                else:
                    st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask a question about the video..."):
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Get response from system
        with st.chat_message("assistant"):
            # Search for relevant information
            results, used_retrievers = search_with_retrievers(prompt, top_k=top_k)
            
            if results:
                st.markdown(f"I found {len(results)} relevant segments in the video:")
                display_results(results, prompt)
                
                # Add assistant message with results to history
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": f"I found {len(results)} relevant segments in the video:",
                    "results": results
                })
            else:
                no_answer_message = (
                    "I couldn't find specific information about that in the video content. "
                    "Please try rephrasing your question or ask about something else."
                )
                st.info(no_answer_message)
                
                # Add assistant message to history
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": no_answer_message
                })

if __name__ == "__main__":
    main()
