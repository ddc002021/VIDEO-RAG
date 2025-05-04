CONFIG_PATH = "utils/config.yaml"

DEFAULT_STT_MODEL_NAME = "base"
DEFAULT_SEGMENTATION_METHOD = "word_based_sentence"
DEFAULT_KEYFRAME_INTERVAL_SEC = 10
DEFAULT_MAX_DURATION_SEC = 10.0
DEFAULT_EVAL_K = [1, 3, 5]
DEFAULT_TIMESTAMP_TOLERANCE_SEC = 5.0
DEFAULT_LATENCY_NUM_RUNS = 50
DEFAULT_TOP_K_RESULTS = 5
DEFAULT_DB_DISTANCE_METRIC = "cosine"

PIPELINE_STEPS = [
    {"name": "Data Preprocessing", "script": "run_preprocessing.py"},
    {"name": "Embedding Generation", "script": "run_embedding.py"},
    {"name": "Index Building", "script": "run_index.py"},
    {"name": "Evaluation", "script": "run_evaluation.py"}
]
