########################################################################
################################## SETUP ###############################
########################################################################

import logging
import coloredlogs
import os
import sys
from dotenv import load_dotenv

coloredlogs.install(level='INFO', fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

try:
    from utils.constants import CONFIG_PATH
    from core.retrieval import FAISSRetriever, TFIDFRetriever, BM25Retriever, PGVectorRetriever
    from utils.utils import load_config, load_json_data, load_embeddings_as_numpy
except ImportError as e:
    logging.error(f"Error importing modules: {e}")
    sys.exit(1)

########################################################################
############################# MAIN BLOCK ###############################
########################################################################

def main():
    logging.info("=== Starting Index Building Pipeline ===")
    config = load_config(CONFIG_PATH)
    if config is None:
        logging.critical("Failed to load configuration. Exiting.")
        return

    # Ensure base directories exist
    data_dir = config.get('data_dir', 'data')
    index_dir = config.get('index_dir', 'indexes')

    faiss_dir = os.path.join(index_dir, "faiss")
    lexical_dir = os.path.join(index_dir, "lexical")
    os.makedirs(faiss_dir, exist_ok=True)
    os.makedirs(lexical_dir, exist_ok=True)

    # --- Load Input Data ---
    logging.info("=== Loading Input Data ===")
    transcript_path = os.path.join(data_dir, "transcripts", "segmented_transcript.json")
    text_embeddings_path = os.path.join(data_dir, "embeddings", "text_embeddings.json")
    image_embeddings_path = os.path.join(data_dir, "embeddings", "image_embeddings.json")

    segments = load_json_data(transcript_path)
    text_embeddings_data, text_emb_dim = load_embeddings_as_numpy(text_embeddings_path)
    image_embeddings_data, image_emb_dim = load_embeddings_as_numpy(image_embeddings_path)

    if segments is None:
        logging.error("Segmented transcript data is required for lexical indexes. Exiting.")
        return
    if text_embeddings_data is None:
        logging.warning("Text embeddings data failed to load. Exiting.")
        return
    if image_embeddings_data is None:
        logging.warning("Image embeddings data failed to load. Exiting.")
        return

    # --- Build FAISS Indexes ---
    logging.info("=== Building FAISS Indexes ===")
    # Text Index
    if text_emb_dim > 0:
        faiss_text_index_path = os.path.join(faiss_dir, "text_embeddings.index")
        logging.info(f"Building FAISS Text Index (Dim: {text_emb_dim}) at {faiss_text_index_path}...")
        try:
            faiss_text_retriever = FAISSRetriever(embedding_dim=text_emb_dim, index_path=faiss_text_index_path)
            faiss_text_retriever.build_index(text_embeddings_data, data_type='text')
            logging.info("FAISS Text Index built successfully.")
        except Exception as e:
            logging.error(f"Failed to build FAISS Text Index: {e}", exc_info=True)
            return
    else:
        logging.warning("Building FAISS Text Index failed due to zero dimension.")
        return

    # Image Index
    if image_emb_dim > 0:
        faiss_image_index_path = os.path.join(faiss_dir, "image_embeddings.index")
        logging.info(f"Building FAISS Image Index (Dim: {image_emb_dim}) at {faiss_image_index_path}...")
        try:
            faiss_image_retriever = FAISSRetriever(embedding_dim=image_emb_dim, index_path=faiss_image_index_path)
            faiss_image_retriever.build_index(image_embeddings_data, data_type='image')
            logging.info("FAISS Image Index built successfully.")
        except Exception as e:
            logging.error(f"Failed to build FAISS Image Index: {e}", exc_info=True)
            return
    else:
        logging.warning("Building FAISS Image Index failed due to zero dimension.")
        return

    # --- Build Lexical Indexes ---
    logging.info("=== Building Lexical Indexes ===")

    # TF-IDF Index
    logging.info("Building TF-IDF Index...")
    tfidf_index_prefix = os.path.join(lexical_dir, "tfidf")
    try:
        tfidf_retriever = TFIDFRetriever(index_path_prefix=tfidf_index_prefix)
        tfidf_retriever.build_index(segments)
        logging.info(f"TF-IDF Index built successfully (Prefix: {tfidf_index_prefix}).")
    except Exception as e:
        logging.error(f"Failed to build TF-IDF Index: {e}", exc_info=True)
        return

    # BM25 Index
    logging.info("Building BM25 Index...")
    bm25_index_path = os.path.join(lexical_dir, "bm25_index.pkl")
    try:
        bm25_retriever = BM25Retriever(index_path=bm25_index_path)
        bm25_retriever.build_index(segments)
        logging.info(f"BM25 Index built successfully at {bm25_index_path}.")
    except Exception as e:
        logging.error(f"Failed to build BM25 Index: {e}", exc_info=True)
        return
    
    # --- Populate pgvector Database ---
    logging.info("=== Setting up and Populating pgvector Database ===")
    load_dotenv()
    db_config = {
        "dbname": os.getenv("DB_NAME"), "user": os.getenv("DB_USER"),
        "password": os.getenv("DB_PASSWORD"), "host": os.getenv("DB_HOST"),
        "port": os.getenv("DB_PORT")
    }

    text_table = config.get('db_text_table_name', 'text_embeddings')
    image_table = config.get('db_image_table_name', 'image_embeddings')
    db_distance_metric = config.get('db_distance_metric', 'cosine')

    if not all(val is not None for val in db_config.values()):
        logging.error("Database configuration missing in environment variables (.env file).")
        return
    else:
        pg_retriever = None
        try:
            pg_retriever = PGVectorRetriever(
                db_params=db_config,
                text_table_name=text_table,
                image_table_name=image_table,
                text_embedding_dim=text_emb_dim,
                image_embedding_dim=image_emb_dim
            )

            # Setup database (creates both tables and their indexes)
            logging.info(f"Setting up pgvector tables '{text_table}' & '{image_table}' with metric '{db_distance_metric}'...")
            pg_retriever.setup_database(distance_metric=db_distance_metric)

            # Populate text table
            if text_embeddings_data:
                logging.info(f"Populating '{text_table}' with Text Embeddings...")
                pg_retriever.build_index(text_embeddings_data, content_type='text')
                logging.info(f"'{text_table}' populated with Text Embeddings.")
            else:
                logging.warning("Skipping pgvector Text population due to missing data.")

            # Populate image table
            if image_embeddings_data:
                logging.info(f"Populating '{image_table}' with Image Embeddings...")
                pg_retriever.build_index(image_embeddings_data, content_type='image')
                logging.info(f"'{image_table}' populated with Image Embeddings.")
            else:
                logging.warning("Skipping pgvector Image population due to missing data.")

        except Exception as e:
            logging.error(f"Failed during pgvector setup or population: {e}", exc_info=True)
        finally:
            if pg_retriever:
                pg_retriever.close_connection()
                logging.info("pgvector database connection closed.")


    logging.info("=== Index Building Pipeline Finished ===")

if __name__ == "__main__":
    main()