import os
import logging
import coloredlogs
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple, Optional

import pickle
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib 
from scipy.sparse import save_npz, load_npz
import nltk
from nltk.corpus import stopwords
from rank_bm25 import BM25Okapi
import psycopg2
import psycopg2.extras
from pgvector.psycopg2 import register_vector

coloredlogs.install(level='INFO', fmt='%(asctime)s - %(filename)s - %(funcName)s - %(levelname)s - %(message)s')

# ------------------------------------ Base Retriever Class ------------------------------------

class Retriever(ABC):
    """Abstract base class for all retrieval methods."""

    @abstractmethod
    def build_index(self, data: List[Dict[str, Any]], **kwargs):
        """Builds or loads the index for searching."""
        pass

    @abstractmethod
    def search(self, query: str, top_k: int = 5, **kwargs) -> List[Tuple[str, float]]:
        """
        Performs a search against the index.

        Returns:
            A list of tuples, where each tuple contains (item_id, score).
            'item_id' corresponds to the segment or frame ID.
        """
        pass

# ------------------------------------ Semantic Retrievers ------------------------------------

class FAISSRetriever(Retriever):
    """Retrieves using FAISS for semantic search."""
    def __init__(self, embedding_dim: int, index_path: str = "indexes/faiss/vector.index"):
        self.embedding_dim = embedding_dim
        self.index_path = index_path
        self.metadata_path = os.path.splitext(index_path)[0] + "_meta.pkl"
        self.index = None
        self.metadata = [] # This will be populated by load_index

    def build_index(self, embedded_data: List[Dict[str, Any]], data_type: str = 'text'):
        """
        Builds a FAISS index from embeddings and saves both index and metadata.

        Args:
            embedded_data: List of dictionaries containing 'id', 'embedding', and other metadata.
                           Embeddings should be numpy arrays here.
            data_type: 'text' or 'image', used for logging.
        """
        if not embedded_data:
            logging.error("No embedded data provided to build FAISS index.")
            return

        # Extract embeddings making sure they are numpy arrays
        try:
             embeddings = np.array([item['embedding'] for item in embedded_data]).astype('float32')
             logging.info(f"Embeddings shape: {embeddings.shape}")
             if embeddings.ndim != 2:
                  raise ValueError("Embeddings array is not 2D.")
        except KeyError:
             logging.error("Items in embedded_data must have an 'embedding' key.")
             return
        except Exception as e:
             logging.error(f"Error processing embeddings: {e}")
             return

        if embeddings.shape[1] != self.embedding_dim:
            raise ValueError(f"Embedding dimension mismatch. Expected {self.embedding_dim}, got {embeddings.shape[1]}")

        logging.info(f"Building FAISS index ({data_type}) with {len(embeddings)} vectors.")
        
        # Normalize vectors for cosine similarity
        faiss.normalize_L2(embeddings)
        
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        
        self.index.add(embeddings)
        logging.info(f"FAISS index built. Total vectors: {self.index.ntotal}")
        
        # Prepare metadata for saving (strip out the large embeddings to save space)
        metadata_to_save = []
        for item in embedded_data:
            meta_copy = item.copy()
            del meta_copy['embedding'] # Don't save embedding in metadata file
            metadata_to_save.append(meta_copy)

        # --- Save Index and Metadata ---
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.index_path), exist_ok=True)

            # Save FAISS index
            faiss.write_index(self.index, self.index_path)
            logging.info(f"FAISS index saved to {self.index_path}")

            # Save metadata using pickle
            with open(self.metadata_path, 'wb') as f_meta:
                pickle.dump(metadata_to_save, f_meta)
            logging.info(f"Metadata saved to {self.metadata_path}")

            # Keep metadata in memory
            self.metadata = metadata_to_save

        except Exception as e:
             logging.error(f"Error saving FAISS index or metadata: {e}")


    def load_index(self):
         """Loads the FAISS index and associated metadata from disk."""
         if os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
              try:
                  logging.info(f"Loading FAISS index from {self.index_path}")
                  self.index = faiss.read_index(self.index_path)
                  logging.info(f"FAISS index loaded. Total vectors: {self.index.ntotal}")

                  # Load metadata using pickle
                  logging.info(f"Loading metadata from {self.metadata_path}")
                  with open(self.metadata_path, 'rb') as f_meta:
                      self.metadata = pickle.load(f_meta)
                  logging.info(f"Metadata loaded. Number of items: {len(self.metadata)}")

                  # Sanity check
                  if self.index.ntotal != len(self.metadata):
                       logging.error(f"Index size ({self.index.ntotal}) mismatch with metadata size ({len(self.metadata)}). Check files.")
                       self.index = None
                       self.metadata = []
                       return False # Indicate loading failed

                  return True # Indicate loading succeeded

              except Exception as e:
                  logging.error(f"Error loading FAISS index or metadata: {e}")
                  self.index = None
                  self.metadata = []
                  return False
         else:
              if not os.path.exists(self.index_path):
                   logging.error(f"FAISS index file not found at {self.index_path}")
              if not os.path.exists(self.metadata_path):
                   logging.error(f"Metadata file not found at {self.metadata_path}")
              self.index = None
              self.metadata = []
              return False

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[Dict[str, Any], float]]: # Return Dict for meta
        """
        Searches the FAISS index using a query embedding. Requires index and metadata to be loaded.
        """
        # Try loading if index or metadata is missing
        if self.index is None or not self.metadata:
           logging.warning("Index or metadata not loaded. Attempting to load...")
           if not self.load_index(): # load_index now returns True/False
               logging.error("Failed to load index/metadata. Cannot search.")
               return []

        if self.index.ntotal == 0:
            logging.warning("FAISS index is empty. Cannot search.")
            return []
        if not self.metadata: # Double check after loading attempt
             logging.error("Metadata is empty even after loading attempt. Cannot search.")
             return []

        if query_embedding.ndim == 1:
            query_embedding = np.expand_dims(query_embedding, axis=0)

        query_embedding = query_embedding.astype('float32')
        
        # Normalize query vector for cosine similarity
        faiss.normalize_L2(query_embedding)

        try:
            distances, indices = self.index.search(query_embedding, top_k)
        except Exception as e:
            logging.error(f"Error during FAISS search: {e}")
            return []

        results = []
        if len(indices) > 0 and len(distances) > 0:
            for i, idx in enumerate(indices[0]):
                if idx != -1: # Check for valid index
                    if 0 <= idx < len(self.metadata): # Check bounds explicitly
                        meta = self.metadata[idx] # Get the corresponding metadata dict
                        score = float(distances[0][i])
                        results.append((meta, score))
                    else:
                        # This shouldn't happen if index size matches metadata size on load
                        logging.warning(f"Search returned index {idx} which is out of bounds for loaded metadata (size {len(self.metadata)}).")

        # Sort results by score (descending for cosine similarity/inner product)
        results.sort(key=lambda x: x[1], reverse=True)
        return results

class PGVectorRetriever(Retriever):
    """
    Retrieves using PostgreSQL with pgvector extension.
    Handles separate tables for text and image embeddings if dimensions differ.
    """
    def __init__(self,
                 db_params: Dict[str, str],
                 text_table_name: str = "text_embeddings",
                 image_table_name: str = "image_embeddings",
                 text_embedding_dim: Optional[int] = None,
                 image_embedding_dim: Optional[int] = None):
        """
        Args:
            db_params: Dictionary with connection parameters.
            text_table_name: Name for the text embeddings table.
            image_table_name: Name for the image embeddings table.
            text_embedding_dim: Dimension of text vectors (required if inserting text).
            image_embedding_dim: Dimension of image vectors (required if inserting images).
        """
        if text_embedding_dim is None and image_embedding_dim is None:
            raise ValueError("At least one embedding dimension (text or image) must be provided.")

        self.db_params = db_params
        self.text_table_name = text_table_name
        self.image_table_name = image_table_name
        self.text_embedding_dim = text_embedding_dim
        self.image_embedding_dim = image_embedding_dim
        self._conn = None

    def _get_connection(self):
        if self._conn is None or self._conn.closed != 0:
            try:
                logging.info(f"Connecting to PostgreSQL database '{self.db_params.get('dbname')}'...")
                self._conn = psycopg2.connect(**self.db_params)
                register_vector(self._conn)
                logging.info("Database connection successful.")
            except psycopg2.DatabaseError as e:
                logging.error(f"Database connection error: {e}")
                self._conn = None
                raise
        return self._conn

    def close_connection(self):
        if self._conn and self._conn.closed == 0:
            self._conn.close()
            logging.info("Database connection closed.")
            self._conn = None

    def _create_table_and_indexes(self, cur, table_name: str, embedding_dim: int, distance_metric: str):
        """Helper function to create a table and its indexes."""
        if embedding_dim <= 0:
             logging.error(f"Invalid embedding dimension ({embedding_dim}) for table {table_name}. Skipping creation.")
             return

        # Determine vector operator based on metric
        vector_ops_map = {'cosine': 'vector_cosine_ops', 'l2': 'vector_l2_ops', 'inner_product': 'vector_ip_ops'}
        vector_ops = vector_ops_map.get(distance_metric, 'vector_cosine_ops')
        if distance_metric not in vector_ops_map:
            logging.warning(f"Unsupported distance metric: {distance_metric}. Defaulting to cosine.")

        # Create table
        create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            id VARCHAR(255) PRIMARY KEY,
            text TEXT NULL,                 -- For text embeddings
            frame_path VARCHAR(512) NULL,   -- For image embeddings
            start_time REAL NULL,           -- For text segments
            end_time REAL NULL,             -- For text segments
            timestamp_sec REAL NULL,        -- For image keyframes
            embedding VECTOR({embedding_dim}) NOT NULL
        );
        """
        cur.execute(create_table_sql)
        logging.info(f"Table '{table_name}' checked/created with embedding dim {embedding_dim}.")

        # --- Create HNSW Index ---
        hnsw_index_name = f"{table_name}_hnsw_{distance_metric}_idx"
        cur.execute("SELECT indexname FROM pg_indexes WHERE indexname = %s;", (hnsw_index_name,))
        if not cur.fetchone():
            m = 16
            ef_construction = 64
            create_hnsw_sql = f"""
            CREATE INDEX {hnsw_index_name} ON {table_name}
            USING hnsw (embedding {vector_ops})
            WITH (m = {m}, ef_construction = {ef_construction});
            """
            logging.info(f"Creating HNSW index '{hnsw_index_name}' on {table_name}...")
            cur.execute(create_hnsw_sql)
            logging.info(f"HNSW index '{hnsw_index_name}' created.")
        else:
             logging.info(f"HNSW index '{hnsw_index_name}' already exists.")

        # --- Create IVFFlat Index ---
        ivfflat_index_name = f"{table_name}_ivfflat_{distance_metric}_idx"
        cur.execute("SELECT indexname FROM pg_indexes WHERE indexname = %s;", (ivfflat_index_name,))
        if not cur.fetchone():
             cur.execute(f"SELECT count(*) FROM {table_name};")
             row_count = cur.fetchone()[0]
             if row_count < 10000:
                  lists_count = max(4, int(np.sqrt(row_count)) // 2 if row_count > 0 else 4)
             else:
                  lists_count = max(10, int(np.sqrt(row_count)))

             logging.info(f"Estimated row count for IVFFlat lists on {table_name}: {row_count}. Setting lists = {lists_count}.")
             create_ivfflat_sql = f"""
             CREATE INDEX {ivfflat_index_name} ON {table_name}
             USING ivfflat (embedding {vector_ops})
             WITH (lists = {lists_count});
             """
             logging.info(f"Creating IVFFlat index '{ivfflat_index_name}' on {table_name}...")
             cur.execute(create_ivfflat_sql)
             logging.info(f"IVFFlat index '{ivfflat_index_name}' created.")
        else:
             logging.info(f"IVFFlat index '{ivfflat_index_name}' already exists.")


    def setup_database(self, distance_metric: str = 'cosine'):
        """
        Creates separate tables and indexes for text and image embeddings if dimensions are provided.
        """
        conn = self._get_connection()
        cur = conn.cursor()
        try:
            # Ensure pgvector extension exists
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            logging.info("pgvector extension checked/created.")

            # Setup text table if dimension is provided
            if self.text_embedding_dim and self.text_embedding_dim > 0:
                self._create_table_and_indexes(cur, self.text_table_name, self.text_embedding_dim, distance_metric)
            else:
                 logging.info(f"Skipping setup for text table '{self.text_table_name}' as dimension is not provided or invalid.")

            # Setup image table if dimension is provided
            if self.image_embedding_dim and self.image_embedding_dim > 0:
                 self._create_table_and_indexes(cur, self.image_table_name, self.image_embedding_dim, distance_metric)
            else:
                 logging.info(f"Skipping setup for image table '{self.image_table_name}' as dimension is not provided or invalid.")

            conn.commit()
        except psycopg2.Error as e:
            logging.error(f"Database setup error: {e}")
            conn.rollback()
        finally:
            cur.close()

    def build_index(self, embedded_data: List[Dict[str, Any]], content_type: str):
        """Inserts embedded data into the appropriate database table based on content_type."""
        if not embedded_data:
            logging.warning(f"No embedded data provided for {content_type} to insert into DB.")
            return

        if content_type == 'text':
            table_name = self.text_table_name
            expected_dim = self.text_embedding_dim
            if not expected_dim or expected_dim <= 0:
                 logging.error("Text embedding dimension not configured correctly. Cannot insert text data.")
                 return
        elif content_type == 'image':
            table_name = self.image_table_name
            expected_dim = self.image_embedding_dim
            if not expected_dim or expected_dim <= 0:
                 logging.error("Image embedding dimension not configured correctly. Cannot insert image data.")
                 return
        else:
            logging.error(f"Invalid content_type '{content_type}'. Use 'text' or 'image'.")
            return

        conn = self._get_connection()
        cur = conn.cursor()
        insert_sql = f"""
        INSERT INTO {table_name} (
            id, text, frame_path, start_time, end_time, timestamp_sec, embedding
        ) VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (id) DO UPDATE SET
            text = EXCLUDED.text,
            frame_path = EXCLUDED.frame_path,
            start_time = EXCLUDED.start_time,
            end_time = EXCLUDED.end_time,
            timestamp_sec = EXCLUDED.timestamp_sec,
            embedding = EXCLUDED.embedding;
        """
        data_to_insert = []
        try:
            for item in embedded_data:
                embedding_array = np.array(item['embedding'])
                if embedding_array.shape[0] != expected_dim:
                    logging.warning(f"Item ID {item.get('id', 'N/A')} has dimension {embedding_array.shape[0]}, expected {expected_dim} for {content_type}. Skipping.")
                    continue

                # Prepare tuple based on content type
                data_tuple = (
                    item.get('id'),
                    item.get('text') if content_type == 'text' else None,
                    item.get('frame_path') if content_type == 'image' else None,
                    item.get('start') or item.get('start_time'),  # Keep start_time for both text and images
                    item.get('end') or item.get('end_time'),  # Keep end_time for both text and images
                    item.get('timestamp_sec') if content_type == 'image' else None,
                    embedding_array
                )
                data_to_insert.append(data_tuple)

            if data_to_insert:
                logging.info(f"Inserting/updating {len(data_to_insert)} {content_type} embeddings into '{table_name}'...")
                psycopg2.extras.execute_batch(cur, insert_sql, data_to_insert)
                conn.commit()
                logging.info(f"Data insertion into '{table_name}' complete.")
            else:
                logging.warning(f"No valid data tuples formed for insertion into '{table_name}'.")

        except psycopg2.Error as e:
            logging.error(f"Database insertion error for {table_name}: {e}")
            conn.rollback()
        except Exception as e:
            logging.error(f"Unexpected error during data preparation/insertion for {table_name}: {e}")
            conn.rollback()
        finally:
            cur.close()

    def search(self,
               query_embedding: np.ndarray,
               top_k: int = 5,
               content_type: str = 'text',
               distance_metric: str = 'cosine'
               ) -> List[Tuple[Dict[str, Any], float]]:
        """Searches the appropriate database table based on content_type."""

        if content_type == 'text':
            table_name = self.text_table_name
            expected_dim = self.text_embedding_dim
        elif content_type == 'image':
            table_name = self.image_table_name
            expected_dim = self.image_embedding_dim
        else:
            logging.error(f"Invalid content_type '{content_type}' for search.")
            return []

        if not expected_dim or expected_dim <= 0:
             logging.error(f"Embedding dimension for {content_type} not configured correctly. Cannot search.")
             return []

        if query_embedding.ndim == 1:
             query_embedding = np.expand_dims(query_embedding, axis=0)

        if query_embedding.shape[1] != expected_dim:
             logging.error(f"Query embedding dimension ({query_embedding.shape[1]}) does not match expected dimension ({expected_dim}) for {content_type} search.")
             return []

        conn = self._get_connection()
        if conn is None: # Ensure connection is valid before proceeding
             logging.error("Cannot search, database connection is not available.")
             return []

        cur = None # Define cursor outside try block for finally clause
        results = []
        try:
            # Use DictCursor to get results as dictionaries
            cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)

            # --- Alternative Vector String Formatting ---
            # Extract the first (and only) 1D embedding vector
            query_vector_1d = query_embedding[0]
            # Use np.array2string for robust formatting, remove extra spaces/newlines
            query_vector_str = np.array2string(query_vector_1d, separator=',', suppress_small=True, max_line_width=np.inf)
            # Ensure it's exactly in the format '[num1,num2,...]' without outer brackets if array2string adds them
            query_vector_str = query_vector_str.replace('\n', '').replace(' ', '')
            if not query_vector_str.startswith('['): query_vector_str = '[' + query_vector_str
            if not query_vector_str.endswith(']'): query_vector_str = query_vector_str + ']'
            # --- End Alternative Formatting ---

            # Determine operator based on metric
            op_map = {'cosine': ('<=>', 'ASC'), 'l2': ('<->', 'ASC'), 'inner_product': ('<#>', 'DESC')}
            distance_op, order_by_dir = op_map.get(distance_metric, ('<=>', 'ASC')) # Default to cosine

            # Build query parts
            select_clause = f"SELECT id, text, frame_path, start_time, end_time, timestamp_sec, (embedding {distance_op} %s) AS distance"
            from_clause = f"FROM {table_name}"
            order_clause = f"ORDER BY distance {order_by_dir}" # Order by calculated distance alias
            limit_clause = "LIMIT %s"
            query_params = [query_vector_str] # Param for distance calculation

            final_query = f"{select_clause} {from_clause} {order_clause} {limit_clause};"
            query_params.append(top_k) # Param for LIMIT

            # Execute the query
            logging.debug(f"Executing pgvector query on {table_name}: {cur.mogrify(final_query, query_params).decode()}") # Log the query for debugging
            cur.execute(final_query, query_params)
            rows = cur.fetchall()

            for row in rows:
                result_dict = dict(row)
                score = result_dict.pop('distance')
                results.append((result_dict, score))

        except psycopg2.Error as e:
            logging.error(f"Database search error on table {table_name}: {e}")
            # --- ADD ROLLBACK ---
            if conn and conn.closed == 0:
                 try:
                      conn.rollback() # Rollback the failed transaction
                      logging.info(f"Transaction rolled back on table {table_name} due to search error.")
                 except psycopg2.Error as rb_err:
                      logging.error(f"Error during rollback: {rb_err}")
            # --- END ROLLBACK ---
        except Exception as e:
             logging.error(f"Unexpected error during pgvector search on {table_name}: {e}", exc_info=True)
             # Also attempt rollback on unexpected errors
             if conn and conn.closed == 0:
                  try:
                       conn.rollback()
                       logging.info(f"Transaction rolled back on table {table_name} due to unexpected search error.")
                  except psycopg2.Error as rb_err:
                       logging.error(f"Error during rollback: {rb_err}")
        finally:
            if cur and not cur.closed:
                 cur.close()
            # Do NOT close the connection here, let the calling script manage it

        return results

# ------------------------------------ Lexical Retrievers ------------------------------------

class TFIDFRetriever(Retriever):
    """Retrieves using TF-IDF scores."""
    def __init__(self, index_path_prefix: str = "indexes/lexical/tfidf"):
        # Define paths for all components based on the prefix
        self.vectorizer_path = f"{index_path_prefix}_vectorizer.pkl"
        self.matrix_path = f"{index_path_prefix}_matrix.npz"
        self.metadata_path = f"{index_path_prefix}_metadata.pkl" # Path for metadata
        self.vectorizer = None
        self.tfidf_matrix = None
        self.metadata = [] # Will be populated by load_index

    def build_index(self, segments: List[Dict[str, Any]]):
        """Builds TF-IDF vectorizer and matrix, and saves all components."""
        if not segments:
            logging.error("No segments provided for TF-IDF index.")
            return

        metadata_to_save = []
        texts = []
        for seg in segments:
            if seg.get('text'):
                texts.append(seg['text'])
                meta_copy = {
                    'id': seg.get('id'),
                    'start': seg.get('start'),
                    'end': seg.get('end')
                }
                metadata_to_save.append(meta_copy)

        if not texts:
            logging.error("No actual text content found in segments for TF-IDF.")
            return

        logging.info(f"Building TF-IDF index for {len(texts)} documents...")
        self.vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)
        logging.info(f"TF-IDF matrix shape: {self.tfidf_matrix.shape}")

        # --- Save Index Components ---
        try:
            os.makedirs(os.path.dirname(self.vectorizer_path), exist_ok=True)

            # Save vectorizer
            joblib.dump(self.vectorizer, self.vectorizer_path)
            logging.info(f"TF-IDF vectorizer saved to {self.vectorizer_path}")

            # Save sparse matrix
            save_npz(self.matrix_path, self.tfidf_matrix)
            logging.info(f"TF-IDF matrix saved to {self.matrix_path}")

            # Save metadata
            with open(self.metadata_path, 'wb') as f_meta:
                pickle.dump(metadata_to_save, f_meta)
            logging.info(f"Metadata saved to {self.metadata_path}")

            self.metadata = metadata_to_save

        except Exception as e:
            logging.error(f"Error saving TF-IDF index components: {e}")


    def load_index(self):
        """Loads TF-IDF vectorizer, matrix, and associated metadata."""
        if (os.path.exists(self.vectorizer_path) and
            os.path.exists(self.matrix_path) and
            os.path.exists(self.metadata_path)):
            try:
                logging.info(f"Loading TF-IDF index components from prefix {os.path.dirname(self.vectorizer_path)}...")
                # Load vectorizer
                self.vectorizer = joblib.load(self.vectorizer_path)
                logging.info("TF-IDF vectorizer loaded.")

                # Load sparse matrix
                self.tfidf_matrix = load_npz(self.matrix_path)
                logging.info("TF-IDF matrix loaded.")

                # Load metadata
                with open(self.metadata_path, 'rb') as f_meta:
                    self.metadata = pickle.load(f_meta)
                logging.info(f"Metadata loaded. Number of items: {len(self.metadata)}")

                # Sanity check
                if self.tfidf_matrix.shape[0] != len(self.metadata):
                     logging.error(f"Matrix row count ({self.tfidf_matrix.shape[0]}) mismatch with metadata size ({len(self.metadata)}). Check files.")
                     self.vectorizer = None
                     self.tfidf_matrix = None
                     self.metadata = []
                     return False # Indicate failure

                logging.info("TF-IDF index loaded successfully.")
                return True # Indicate success

            except Exception as e:
                logging.error(f"Error loading TF-IDF index components: {e}")
                self.vectorizer = None
                self.tfidf_matrix = None
                self.metadata = []
                return False
        else:
            logging.error(f"One or more TF-IDF index files not found for prefix {os.path.dirname(self.vectorizer_path)}")
            self.vectorizer = None
            self.tfidf_matrix = None
            self.metadata = []
            return False


    def search(self, query: str, top_k: int = 5) -> List[Tuple[Dict[str, Any], float]]: # Return Dict for meta
        """Searches using TF-IDF cosine similarity. Requires index and metadata to be loaded."""
        # Try loading if index or metadata is missing
        if self.vectorizer is None or self.tfidf_matrix is None or not self.metadata:
            logging.warning("TF-IDF components not loaded. Attempting to load...")
            if not self.load_index():
                 logging.error("Failed to load TF-IDF index/metadata. Cannot search.")
                 return []

        if self.tfidf_matrix.shape[0] == 0:
            logging.warning("TF-IDF matrix is empty.")
            return []
        if not self.metadata: # Double check after loading attempt
             logging.error("Metadata is empty even after loading attempt. Cannot search.")
             return []

        try:
            query_vec = self.vectorizer.transform([query])
            similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        except Exception as e:
             logging.error(f"Error during TF-IDF search: {e}")
             return []

        # Get top_k indices efficiently
        # Handle case where k is larger than the number of documents
        actual_top_k = min(top_k, similarities.shape[0])
        if actual_top_k <= 0:
            return []

        # Use argpartition for efficiency if many documents, then sort the top-k
        # For potentially small N (like segments in one video), simple argsort might be fine too
        # top_k_indices = np.argsort(similarities)[-actual_top_k:][::-1]
        k_th_value = similarities.shape[0] - actual_top_k
        top_k_indices = np.argpartition(similarities, k_th_value)[k_th_value:]
        top_k_indices = top_k_indices[np.argsort(similarities[top_k_indices])][::-1] # Sort descending

        results = []
        for idx in top_k_indices:
            score = float(similarities[idx])
            # Optionally filter out zero or very low scores if desired
            if score > 1e-6: # Check for non-zero similarity
                if 0 <= idx < len(self.metadata):
                    meta = self.metadata[idx] # Get the corresponding metadata dict
                    results.append((meta, score))
                else:
                    logging.warning(f"Search returned index {idx} which is out of bounds for loaded metadata (size {len(self.metadata)}).")

        # Results are already sorted by score (descending)
        return results

def bm25_tokenizer(text):
    """Basic tokenizer for BM25: lowercase, punctuation removal, stopword removal."""
    if text is None:
        return []
    try:
        # Ensure necessary NLTK data is available (download if needed)
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            logging.info("NLTK 'punkt' resource not found. Downloading...")
            nltk.download('punkt', quiet=True)
        try:
            nltk.data.find('tokenizers/punkt_tab')
        except LookupError:
            logging.info("NLTK 'punkt_tab' resource not found. Downloading...")
            nltk.download('punkt_tab', quiet=True)
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            logging.info("NLTK 'stopwords' resource not found. Downloading...")
            nltk.download('stopwords', quiet=True)

        # Perform tokenization
        tokens = nltk.word_tokenize(text.lower())
        stop_words = set(stopwords.words('english'))
        # Keep only alphanumeric tokens that are not stopwords
        return [token for token in tokens if token.isalnum() and token not in stop_words]
    except Exception as e:
        logging.warning(f"NLTK tokenization failed: {e}. Returning empty list for this document.")
        return []

class BM25Retriever(Retriever):
    """Retrieves using BM25 scores."""
    def __init__(self, index_path: str = "indexes/lexical/bm25_index.pkl"):
        self.index_path = index_path
        self.bm25 = None
        self.metadata = []

    def build_index(self, segments: List[Dict[str, Any]]):
        """Builds BM25 index."""
        if not segments:
             logging.error("No segments provided for BM25 index.")
             return

        metadata_to_save = []
        texts = []
        for seg in segments:
            if seg.get('text'):
                texts.append(seg['text'])
                meta_copy = {
                    'id': seg.get('id'),
                    'start': seg.get('start'),
                    'end': seg.get('end')
                }
                metadata_to_save.append(meta_copy)

        if not texts:
            logging.error("No actual text content found in segments for BM25.")
            return

        logging.info(f"Tokenizing {len(texts)} documents for BM25...")
        tokenized_corpus = [bm25_tokenizer(text) for text in texts]

        if not any(tokenized_corpus): 
             logging.warning("Tokenization resulted in empty lists for all documents.")
             return

        logging.info("Building BM25 index...")
        self.bm25 = BM25Okapi(tokenized_corpus)
        logging.info("BM25 index built.")

        # Save index (BM25 object and metadata)
        self.metadata = metadata_to_save
        save_data = {'bm25': self.bm25, 'metadata': self.metadata}

        try:
            os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
            with open(self.index_path, 'wb') as f_out:
                 pickle.dump(save_data, f_out)
            logging.info(f"BM25 index and metadata saved to {self.index_path}")
        except Exception as e:
             logging.error(f"Failed to save BM25 index/metadata: {e}")


    def load_index(self):
        """Loads BM25 index object and associated metadata."""
        if os.path.exists(self.index_path):
            try:
                logging.info(f"Loading BM25 index and metadata from {self.index_path}")
                # Use pickle to load
                with open(self.index_path, 'rb') as f_in:
                     loaded_data = pickle.load(f_in)

                self.bm25 = loaded_data.get('bm25')
                self.metadata = loaded_data.get('metadata', [])

                if self.bm25 is None:
                     logging.error("Loaded data does not contain 'bm25' object.")
                     return False

                # Optional sanity check: Compare corpus count if BM25 object stores it
                if hasattr(self.bm25, 'corpus_size') and self.bm25.corpus_size != len(self.metadata):
                     logging.warning(f"BM25 corpus size ({getattr(self.bm25, 'corpus_size', 'N/A')}) mismatch with loaded metadata size ({len(self.metadata)}).")
                     # Decide if this is critical - often it's okay if metadata is right length

                logging.info(f"BM25 index loaded. Metadata items: {len(self.metadata)}")
                return True
            except Exception as e:
                 logging.error(f"Error loading BM25 index file {self.index_path}: {e}")
                 self.bm25 = None
                 self.metadata = []
                 return False
        else:
            logging.error(f"BM25 index file not found: {self.index_path}")
            self.bm25 = None
            self.metadata = []
            return False


    def search(self, query: str, top_k: int = 5) -> List[Tuple[Dict[str, Any], float]]: # Return Dict for meta
        """Searches using BM25 scores. Requires index and metadata to be loaded."""
        if self.bm25 is None or not self.metadata:
            logging.warning("BM25 index or metadata not loaded. Attempting to load...")
            if not self.load_index():
                logging.error("Failed to load BM25 index/metadata. Cannot search.")
                return []

        if not self.metadata: # Double check after loading attempt
             logging.error("Metadata is empty even after loading attempt. Cannot search.")
             return []

        try:
            tokenized_query = bm25_tokenizer(query)
            if not tokenized_query:
                 logging.warning(f"Query '{query}' resulted in empty tokens after processing. BM25 search might yield no results.")
                 # Still proceed, BM25 handles empty query tokens

            # Check if bm25 object has the get_scores method (basic check)
            if not hasattr(self.bm25, 'get_scores'):
                 logging.error("Loaded BM25 object does not have 'get_scores' method.")
                 return []

            doc_scores = self.bm25.get_scores(tokenized_query)

            if len(doc_scores) != len(self.metadata):
                 logging.error(f"BM25 returned {len(doc_scores)} scores, but metadata has {len(self.metadata)} items. Mismatch!")
                 # This indicates a serious issue, likely during loading/saving
                 return []

        except Exception as e:
             logging.error(f"Error during BM25 score calculation: {e}")
             return []

        # Get top_k indices
        actual_top_k = min(top_k, len(doc_scores))
        if actual_top_k <= 0:
             return []

        # Use argpartition for efficiency
        k_th_value = len(doc_scores) - actual_top_k
        top_k_indices = np.argpartition(doc_scores, k_th_value)[k_th_value:]
        top_k_indices = top_k_indices[np.argsort(doc_scores[top_k_indices])][::-1] # Sort descending

        results = []
        for idx in top_k_indices:
            score = float(doc_scores[idx])
            # BM25 scores can be 0 or negative, filter if needed (often > 0 is desired)
            if score > 1e-6: # Check for non-zero score
                if 0 <= idx < len(self.metadata):
                    meta = self.metadata[idx] # Get the corresponding metadata dict
                    results.append((meta, score))
                else:
                    # This check should ideally never fail if lengths match above
                    logging.warning(f"Search returned index {idx} which is out of bounds for loaded metadata (size {len(self.metadata)}).")

        return results
