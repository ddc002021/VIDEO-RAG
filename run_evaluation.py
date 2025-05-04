########################################################################
################################## SETUP ###############################
########################################################################

import yaml
import json
import os
import sys
import logging
import coloredlogs
from dotenv import load_dotenv

coloredlogs.install(level='INFO', fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

try:
    from utils.constants import CONFIG_PATH
    from utils.utils import load_config, load_json_data, get_embedding_dimensions, parse_timestamp
    from core.embedding import load_text_model, load_vision_model
    from core.retrieval import FAISSRetriever, TFIDFRetriever, BM25Retriever, PGVectorRetriever
    from core.evaluation import evaluate_retrievers, calculate_metrics
except ImportError as e:
    logging.basicConfig(level=logging.ERROR)
    logging.error(f"Error importing core modules: {e}.")
    sys.exit(1)

coloredlogs.install(level='INFO', fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

########################################################################
############################# MAIN BLOCK ###############################
########################################################################

def main():
    logging.info("=== Starting Retrieval Evaluation Pipeline ===")

    logging.info("=== Loading Configuration and Gold Standard Data ===")
    config = load_config(CONFIG_PATH)
    if config is None:
        logging.critical("Failed to load configuration. Exiting.")
        return

    gold_standard_path = config.get('gold_standard_path', 'data/gold_standard.json')
    gold_standard_data = load_json_data(gold_standard_path)
    if gold_standard_data is None:
        logging.critical("Failed to load gold standard data. Exiting.")
        return

    # Parse ground truth timestamps
    for item in gold_standard_data:
        item['ground_truth_seconds'] = [ts for ts_str in item.get('ground_truth_timestamps', []) if (ts := parse_timestamp(ts_str)) is not None]

    # --- Load Embedding Models ---
    logging.info("=== Loading Embedding Models ===")
    text_model_name = config.get('text_embedding_model')
    vision_model_name = config.get('vision_embedding_model')

    if not text_model_name or not vision_model_name:
        logging.critical("Embedding model names not specified in config. Exiting.")
        return

    try:
        text_model = load_text_model(text_model_name)
        vision_model = load_vision_model(vision_model_name)
        if text_model is None or vision_model is None:
            raise ValueError("One or more models failed to load.")
        logging.info("Embedding models loaded successfully.")
    except Exception as e:
         logging.critical(f"Failed to load embedding models: {e}", exc_info=True)
         return

    # --- Load Retrievers and Indexes ---
    logging.info("=== Initializing Retrievers and Loading Indexes ===")
    retrievers = {}

    # Determine embedding dimensions (needed for FAISS/pgvector initialization)
    text_emb_dim, image_emb_dim = get_embedding_dimensions(config)
    logging.info(f"Using Text Dim: {text_emb_dim}, Image Dim: {image_emb_dim} for retriever init.")

    index_dir = config.get('index_dir', 'indexes')
    # FAISS Text
    try:
        faiss_text_index_path = os.path.join(index_dir, "faiss", "text_embeddings.index")
        faiss_text_retriever = FAISSRetriever(embedding_dim=text_emb_dim, index_path=faiss_text_index_path)
        if faiss_text_retriever.load_index():
            retrievers['FAISS_Text'] = faiss_text_retriever
            logging.info("FAISS Text Retriever loaded.")
        else: logging.warning("Failed to load FAISS Text Retriever.")
    except Exception as e: logging.error(f"Error initializing FAISS Text Retriever: {e}")

    # FAISS Image
    try:
        faiss_image_index_path = os.path.join(index_dir, "faiss", "image_embeddings.index")
        faiss_image_retriever = FAISSRetriever(embedding_dim=image_emb_dim, index_path=faiss_image_index_path)
        if faiss_image_retriever.load_index():
            retrievers['FAISS_Image'] = faiss_image_retriever
            logging.info("FAISS Image Retriever loaded.")
        else: logging.warning("Failed to load FAISS Image Retriever.")
    except Exception as e: logging.error(f"Error initializing FAISS Image Retriever: {e}")

    # TF-IDF
    try:
        tfidf_index_prefix = os.path.join(index_dir, "lexical", "tfidf")
        tfidf_retriever = TFIDFRetriever(index_path_prefix=tfidf_index_prefix)
        if tfidf_retriever.load_index():
            retrievers['TF-IDF'] = tfidf_retriever
            logging.info("TF-IDF Retriever loaded.")
        else: logging.warning("Failed to load TF-IDF Retriever.")
    except Exception as e: logging.error(f"Error initializing TF-IDF Retriever: {e}")

    # BM25
    try:
        bm25_index_path = os.path.join(index_dir, "lexical", "bm25_index.pkl")
        bm25_retriever = BM25Retriever(index_path=bm25_index_path)
        if bm25_retriever.load_index():
            retrievers['BM25'] = bm25_retriever
            logging.info("BM25 Retriever loaded.")
        else: logging.warning("Failed to load BM25 Retriever.")
    except Exception as e: logging.error(f"Error initializing BM25 Retriever: {e}")

    # pgvector
    try:
        load_dotenv()
        db_config = {
            "dbname": os.getenv("DB_NAME"), "user": os.getenv("DB_USER"),
            "password": os.getenv("DB_PASSWORD"), "host": os.getenv("DB_HOST"),
            "port": os.getenv("DB_PORT")
        }
      
        if not all(val is not None for val in db_config.values()):
            logging.error("DB config missing in env vars. Skipping pgvector.")
        else:
            pg_text_table = config.get('db_text_table_name', 'text_embeddings')
            pg_image_table = config.get('db_image_table_name', 'image_embeddings')
            
            # Create a single PGVectorRetriever instance that will be shared
            pg_retriever = PGVectorRetriever(
                db_params=db_config,
                text_table_name=pg_text_table, image_table_name=pg_image_table,
                text_embedding_dim=text_emb_dim, image_embedding_dim=image_emb_dim
            )
            # Test connection before adding
            conn = pg_retriever._get_connection()
            if conn:
                # Add as separate text and image retrievers
                retrievers['pgvector_text'] = pg_retriever
                retrievers['pgvector_image'] = pg_retriever  # Same instance, different name
                logging.info("pgvector Text and Image Retrievers initialized (connection tested).")
            else:
                logging.warning("Failed to connect to pgvector database.")
    except Exception as e:
        logging.error(f"Error initializing pgvector Retrievers: {e}")

    if not retrievers:
        logging.critical("No retrievers were loaded successfully. Cannot proceed.")
        return

    logging.info(f"Evaluating retrievers: {list(retrievers.keys())}")

    # --- Run Evaluation Loop ---
    logging.info("=== Running Evaluation Loop ===")

    raw_evaluation_results = evaluate_retrievers(
        gold_standard_data=gold_standard_data,
        retrievers=retrievers,
        text_model=text_model,
        vision_model=vision_model,
        config=config
    )

    # --- Calculate and Display Metrics ---
    logging.info("=== Calculating Overall Metrics ===")
    metrics_df = calculate_metrics(
        evaluation_results=raw_evaluation_results,
        retriever_names=list(retrievers.keys()),
        config=config
    )

    print("\n" + "="*30 + " Aggregated Metrics " + "="*30)
    # Print DataFrame in a readable format (Markdown)
    print(metrics_df.to_markdown(floatfmt=".4f"))
    print("="*80)
    
    # Print explanation of metrics
    print("\nMetric Explanations:")
    print("- MRR (Mean Reciprocal Rank): Higher is better. Rewards retrievers that place correct answers higher in results.")
    print("- Recall@k: Higher is better. Percentage of answerable questions where a correct answer is within top-k results.")
    print("- Unanswerable_Accuracy: Higher is better. Ability to correctly return no results for unanswerable questions.")
    print("- False_Positive_Rate: Lower is better. Percentage of unanswerable questions that wrongly returned results.")
    print("- Avg_Latency_ms: Lower is better. Average query processing time in milliseconds.")
    print("="*80)
    
    # Print relevance threshold information
    thresholds = config.get('relevance_thresholds', {})
    if thresholds:
        print("\nRelevance Thresholds:")
        for retriever, threshold in thresholds.items():
            print(f"- {retriever}: {threshold}")
        print("Note: Results with scores below these thresholds are filtered out.")
        print("="*80)
    
    # Print question statistics
    answerable = sum(1 for item in raw_evaluation_results if item['answerable'])
    unanswerable = len(raw_evaluation_results) - answerable
    print(f"\nQuery Set Statistics: {len(raw_evaluation_results)} total questions")
    print(f"- {answerable} answerable questions ({answerable/len(raw_evaluation_results)*100:.1f}%)")
    print(f"- {unanswerable} unanswerable questions ({unanswerable/len(raw_evaluation_results)*100:.1f}%)")
    print("="*80)

    # --- Cleanup ---
    # Close pgvector connection - since we're using the same instance, we only need to close once
    if 'pgvector_text' in retrievers:
        try:
            retrievers['pgvector_text'].close_connection()
        except Exception as e:
            logging.error(f"Error closing pgvector connection: {e}")

    logging.info("=== Retrieval Evaluation Pipeline Finished ===")

if __name__ == "__main__":
    main()
