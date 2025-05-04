# Multimodal RAG Video Query System

This project implements a Retrieval-Augmented Generation (RAG) system that allows users to query the content of a specific video using natural language. The system leverages both the audio (transcript) and visual (keyframes) information from the video, comparing multiple retrieval techniques.

**Video Used:** Parameterized Complexity of token sliding, token jumping - Amer Mouawad
([Link to video, http://www.youtube.com/watch?v=dARr3lGKwk8])

# ##################################
## Features
# ##################################

* **Multimodal Indexing:** Extracts audio transcripts (using OpenAI Whisper) and video keyframes.
* **Embedding Generation:** Creates vector embeddings for text segments and image frames using open-source models (configurable).
* **Retrieval Methods Implemented:**
    * Semantic Search: FAISS (FlatIP), PostgreSQL with pgvector (HNSW, IVFFlat)
    * Lexical Search: TF-IDF, BM25
* **Evaluation Framework:** Compares retrieval methods using a manually created gold standard dataset (Recall@k, MRR, Unanswerable Accuracy, Average Retreival Time, False Positive Rate).
* **Interactive UI:** A Streamlit application allows users to ask questions and view relevant video timestamps.

# ##################################
## Project Structure
# ##################################
```text
multimodal-rag-video/
├── app/
│   └── streamlit_app.py         # Main Streamlit application
├── core/
│   ├── processing.py            # STT, frame extraction, segmentation
│   ├── embedding.py             # Embedding generation
│   ├── retrieval.py             # Retriever classes (FAISS, TF-IDF, BM25, pgvector)
│   └── evaluation.py            # Evaluation logic and metrics calculation
├── data/
│   ├── video/                   # Downloaded video file
│   ├── audio/                   # Extracted audio file
│   ├── transcripts/             # Raw and segmented transcripts (JSON)
│   ├── keyframes/               # Extracted keyframe images (JPG)
│   ├── embeddings/              # Saved text and image embeddings (JSON)
│   └── gold_standard.json       # Evaluation questions and answers
├── indexes/
│   ├── faiss/                   # Saved FAISS index files
│   └── lexical/                 # Saved TF-IDF/BM25 index files
├── utils/                       # Utilities used by the project
│   ├── config.yaml              # Project configuration (paths, models, params)
│   ├── constants.py             # Project constants (paths and default params)
│   └── utils.py                 # Utility functions
├── .env                         # Environment variables (DB credentials - NOT COMMITTED)
├── .gitignore                   # Specifies intentionally untracked files
├── requirements.txt             # Python package dependencies
├── run_preprocessing.py         # Script to run data loading & processing
├── run_embedding.py             # Script to generate embeddings
├── run_index.py                 # Script to build all search indexes
├── run_evaluation.py            # Script to run evaluation metrics 
└── README.md                    # This file
```

# ##################################
## Setup
# ##################################

1.  **Clone the Repository**

2.  **Create Virtual Environment** (Recommended)

3.  **Install Dependencies:**
    * Make sure you have **Python 3.9+** installed.
    * Install **FFmpeg**: This is required by Whisper and other audio/video libraries. Follow platform-specific instructions (e.g., download from official site, add to PATH). Verify with `ffmpeg -version`.
    * Install Python packages from requirements.txt
    * **NLTK Data:** The first time you run code using BM25, it might attempt to download `punkt` and `stopwords` data from NLTK. Allow the download or run `python -m nltk.downloader punkt stopwords` manually.
    * Download the youtube video and place it in data/video (Name it as complexity_talk.mp4 or update video_filename in config.yaml)
    * Extract the audio and place it in data/audio (Name it as complexity_talk.mp3 or update audio_filename in config.yaml)   

4.  **Setup PostgreSQL & pgvector:**
    * Install PostgreSQL server (version compatible with pgvector, e.g., 15, 16, 17).
    * Install the `pgvector` extension for your PostgreSQL version. See `pgvector` documentation for details.
    * Create a database (e.g., `rag_video_db`).
    * Create a database user (e.g., `rag_user`) with a password and grant it permissions on the database and `public` schema (`USAGE`, `CREATE`).
    * Enable the extension within your database: `CREATE EXTENSION IF NOT EXISTS vector;` (run via psql or pgAdmin).
    * Create a `.env` file in the project root (add it to `.gitignore`!) with your database credentials:
        ```dotenv
        DB_NAME=your_db_name
        DB_USER=your_db_user
        DB_PASSWORD=your_db_password
        DB_HOST=localhost
        DB_PORT=5432
        ```

5.  **Configure `config.yaml`:**
    * Review `config.yaml`.
    * Set the correct `video_url`.
    * Choose your desired open-source `text_embedding_model` and `vision_embedding_model` (Hugging Face identifiers).
    * Adjust other parameters if needed.

# ##################################
## Usage
# ##################################

Execute the following scripts from the project root directory in order:

1.  **Preprocessing:** Transcribes and extracts keyframes.
    ```bash
    python run_preprocessing.py
    ```
    *(Check `data/` subdirectories for output files)*

2.  **Embedding Generation:** Creates text and image embeddings.
    ```bash
    python run_embedding.py
    ```
    *(Check `data/embeddings/` for output JSON files)*

3.  **Index Creation:** Builds FAISS, TF-IDF, BM25, and populates pgvector.
    ```bash
    python run_index.py
    ```
    *(Check `indexes/` subdirectories and your pgvector DB)*

4.  **Evaluation:** Runs gold standard questions against all indexes and calculates metrics.
    ```bash
    python run_evaluation.py
    ```
    *(Observe the printed metrics table)*

# ##################################
## Streamlit App:
# ##################################

To start the interactive web application. Make sure that run_preprocessing.py, run_embedding.py and run_index.py are executed, then run:
    ```bash
    streamlit run app/streamlit_app.py
    ```
*(Access the app via the URL provided in the terminal)*