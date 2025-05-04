import logging
import numpy as np
import pandas as pd
import time
from collections import defaultdict
from typing import List, Dict, Any, Tuple, Optional
from utils.utils import check_match_text, check_match_image
from utils.constants import (
    DEFAULT_EVAL_K,
    DEFAULT_TIMESTAMP_TOLERANCE_SEC,
    DEFAULT_LATENCY_NUM_RUNS,
    DEFAULT_TOP_K_RESULTS,
    DEFAULT_DB_DISTANCE_METRIC
)

def run_single_retrieval(
    query_text: str,
    retriever_name: str,
    retriever: Any, 
    text_model: Any, 
    vision_model: Any, 
    config: Dict[str, Any]
) -> Tuple[List[Tuple[Dict[str, Any], float]], float]:
    """
    Runs a single query against a specific retriever.

    Args:
        query_text: The user's question.
        retriever_name: Name of the retriever (e.g., 'FAISS_Text', 'BM25').
        retriever: The instantiated retriever object.
        text_model: The text embedding model instance.
        vision_model: The vision embedding model instance (CLIP).
        config: The loaded configuration dictionary.

    Returns:
        Tuple of (search_results, latency_seconds)
        - search_results: List of (metadata, score) tuples from the retriever
        - latency_seconds: Time taken in seconds for the search operation
    """
    top_k = config.get('top_k_results', DEFAULT_TOP_K_RESULTS)
    db_distance_metric = config.get('db_distance_metric', DEFAULT_DB_DISTANCE_METRIC)
    num_latency_runs = config.get('latency_num_runs', DEFAULT_LATENCY_NUM_RUNS)

    try:
        if retriever_name in ['FAISS_Text', 'pgvector_text']:
            query_embedding = text_model.encode([query_text], convert_to_numpy=True)
        elif retriever_name in ['FAISS_Image', 'pgvector_image']:
            query_embedding = vision_model.encode([query_text], convert_to_numpy=True)
        
        # First run to get actual results
        start_time = time.time()
        if retriever_name == 'FAISS_Text':
            search_results = retriever.search(query_embedding, top_k=top_k)
        elif retriever_name == 'FAISS_Image':
            search_results = retriever.search(query_embedding, top_k=top_k)
        elif retriever_name == 'pgvector_text':
            search_results = retriever.search(query_embedding, top_k=top_k, content_type='text', distance_metric=db_distance_metric)
        elif retriever_name == 'pgvector_image':
            search_results = retriever.search(query_embedding, top_k=top_k, content_type='image', distance_metric=db_distance_metric)
        elif retriever_name in ['TF-IDF', 'BM25']:
            search_results = retriever.search(query_text, top_k=top_k)
        else:
            logging.warning(f"Retriever type {retriever_name} not recognized for search logic.")
            search_results = []
            return search_results, 0.0
        
        # Record first run time
        first_run_time = time.time() - start_time
        total_time = first_run_time
        
        # Additional runs for more accurate latency measurement (if configured)
        if num_latency_runs > 1:
            for _ in range(num_latency_runs - 1):
                start_time = time.time()
                if retriever_name == 'FAISS_Text':
                    retriever.search(query_embedding, top_k=top_k)
                elif retriever_name == 'FAISS_Image':
                    retriever.search(query_embedding, top_k=top_k)
                elif retriever_name == 'pgvector_text':
                    retriever.search(query_embedding, top_k=top_k, content_type='text', distance_metric=db_distance_metric)
                elif retriever_name == 'pgvector_image':
                    retriever.search(query_embedding, top_k=top_k, content_type='image', distance_metric=db_distance_metric)
                elif retriever_name in ['TF-IDF', 'BM25']:
                    retriever.search(query_text, top_k=top_k)
                total_time += time.time() - start_time
        
        # Calculate average latency
        avg_latency = total_time / num_latency_runs
        return search_results, avg_latency

    except Exception as e:
        logging.error(f"Error during search with {retriever_name} for query '{query_text[:50]}...': {e}", exc_info=True)
        return [], 0.0

def evaluate_retrievers(
    gold_standard_data: List[Dict[str, Any]],
    retrievers: Dict[str, Any],
    text_model: Any,
    vision_model: Any,
    config: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Evaluates multiple retrievers against a gold standard dataset.

    Args:
        gold_standard_data: List of dictionaries (from gold_standard.json, with parsed timestamps).
        retrievers: Dictionary mapping retriever names to their loaded instances.
        text_model: Instantiated text embedding model.
        vision_model: Instantiated vision embedding model (CLIP).
        config: Loaded configuration dictionary.

    Returns:
        List of dictionaries, each containing results for one question across all retrievers.
    """
    evaluation_results = []
    max_k = max(config.get('eval_k', DEFAULT_EVAL_K)) 
    timestamp_tolerance = config.get('timestamp_tolerance_sec', DEFAULT_TIMESTAMP_TOLERANCE_SEC)
    
    # Get relevance thresholds from config
    relevance_thresholds = config.get('relevance_thresholds', {})

    for item in gold_standard_data:
        question_id = item['id']
        query_text = item['question']
        is_answerable = item['answerable']
        ground_truth_secs = item['ground_truth_seconds']

        logging.info(f"Processing Question {question_id}: '{query_text}' (Answerable: {is_answerable}, Ground Truth: {ground_truth_secs})")

        result_row = {'question_id': question_id, 'question': query_text, 'answerable': is_answerable}

        for retriever_name, retriever in retrievers.items():
            logging.debug(f"  Querying {retriever_name}...")

            search_results, latency = run_single_retrieval(
                query_text, retriever_name, retriever, text_model, vision_model, config
            )

            # Apply relevance threshold filtering
            threshold = relevance_thresholds.get(retriever_name)
            relevant_results = []
            
            if threshold is not None:
                # For most retrievers, higher score is better (cosine similarity, BM25, TF-IDF)
                for meta, score in search_results:
                    if score >= threshold:
                        relevant_results.append((meta, score))
                
                logging.debug(f"  {retriever_name}: {len(search_results)} results, {len(relevant_results)} above threshold {threshold}")
                
                # If no results are relevant enough, treat as empty results
                if not relevant_results and search_results:
                    logging.info(f"  {retriever_name}: All {len(search_results)} results filtered out by threshold {threshold} for question {question_id}")
            else:
                # If no threshold configured, use all results
                relevant_results = search_results
                
            # Process results for evaluation metrics
            retrieved_correctly = False
            first_correct_rank = -1

            for rank, (meta, score) in enumerate(relevant_results):
                # Extract timestamp (handle different key names from different retrievers/tables)
                if retriever_name in ['FAISS_Text', 'TF-IDF', 'BM25', 'pgvector_text']:
                    retrieved_ts_start = meta.get('start_time') or meta.get('start')
                    retrieved_ts_end = meta.get('end_time') or meta.get('end')

                    if isinstance(retrieved_ts_start, (int, float)):
                        retrieved_ts_start = float(retrieved_ts_start)
                    else:
                        retrieved_ts_start = None

                    if isinstance(retrieved_ts_end, (int, float)):
                        retrieved_ts_end = float(retrieved_ts_end)
                    else:
                        retrieved_ts_end = None

                    # Check if it's a correct match (only for answerable questions)
                    if is_answerable and not retrieved_correctly:
                        if check_match_text(retrieved_ts_start, retrieved_ts_end, ground_truth_secs):
                            retrieved_correctly = True
                            first_correct_rank = rank + 1

                elif retriever_name in ['FAISS_Image', 'pgvector_image']:
                    retrieved_ts = meta.get('timestamp_sec')
                    # Check for start/end times (scene detection mode)
                    retrieved_ts_start = meta.get('start_time')
                    retrieved_ts_end = meta.get('end_time')

                    # Convert to float if provided
                    if isinstance(retrieved_ts, (int, float)):
                        retrieved_ts = float(retrieved_ts)
                    else:
                        retrieved_ts = None
                        
                    if isinstance(retrieved_ts_start, (int, float)):
                        retrieved_ts_start = float(retrieved_ts_start) 
                    else:
                        retrieved_ts_start = None
                        
                    if isinstance(retrieved_ts_end, (int, float)):
                        retrieved_ts_end = float(retrieved_ts_end)
                    else:
                        retrieved_ts_end = None

                    if is_answerable and not retrieved_correctly:
                        if check_match_image(retrieved_ts, ground_truth_secs, timestamp_tolerance, 
                                           retrieved_ts_start, retrieved_ts_end):
                            retrieved_correctly = True
                            first_correct_rank = rank + 1

            # Store metrics for this retriever and question
            # Use filtered results for determining if anything was found
            result_row[f'{retriever_name}_found_any'] = len(relevant_results) > 0
            result_row[f'{retriever_name}_retrieved_correctly'] = retrieved_correctly
            result_row[f'{retriever_name}_first_correct_rank'] = first_correct_rank
            result_row[f'{retriever_name}_latency'] = latency
            
            # Store additional info for debugging
            if not is_answerable:
                result_row[f'{retriever_name}_raw_result_count'] = len(search_results)
                result_row[f'{retriever_name}_threshold'] = threshold
                
                # Log if this is a correct rejection after thresholding
                if len(search_results) > 0 and len(relevant_results) == 0:
                    logging.info(f"  {retriever_name}: Correctly rejected unanswerable question {question_id} after threshold filtering")

        evaluation_results.append(result_row)

    return evaluation_results


def calculate_metrics(
    evaluation_results: List[Dict[str, Any]],
    retriever_names: List[str],
    config: Dict[str, Any]
) -> pd.DataFrame:
    """
    Calculates aggregated metrics (MRR, Recall@k, Unanswerable Accuracy, Latency)
    from per-question evaluation results.

    Args:
        evaluation_results: List of dictionaries returned by evaluate_retrievers.
        retriever_names: List of retriever names that were evaluated.
        config: Loaded configuration dictionary.

    Returns:
        pandas DataFrame containing the aggregated metrics for each retriever.
    """
    metrics = defaultdict(lambda: defaultdict(float))
    eval_k = config.get('eval_k', DEFAULT_EVAL_K) 

    total_questions = len(evaluation_results)
    total_answerable = sum(1 for item in evaluation_results if item['answerable'])
    total_unanswerable = total_questions - total_answerable

    for retriever_name in retriever_names:
        correct_hits_at_k = defaultdict(int)
        sum_reciprocal_rank = 0.0
        unanswerable_correctly_rejected = 0
        unanswerable_false_positives = 0
        processed_answerable = 0
        processed_unanswerable = 0
        total_latency = 0.0
        latency_count = 0

        for result in evaluation_results:
            is_answerable = result['answerable']
            # Check if results exist for this retriever (might have failed)
            rank_key = f'{retriever_name}_first_correct_rank'
            found_any_key = f'{retriever_name}_found_any'
            latency_key = f'{retriever_name}_latency'

            if rank_key not in result: # Skip if retriever failed for this question
                continue

            rank = result[rank_key]
            found_any = result[found_any_key]
            
            # Add latency tracking
            if latency_key in result and result[latency_key] > 0:
                total_latency += result[latency_key]
                latency_count += 1

            if is_answerable:
                processed_answerable += 1
                if rank != -1: # Correct answer was found
                    sum_reciprocal_rank += 1.0 / rank
                    for k in eval_k:
                        if rank <= k:
                            correct_hits_at_k[k] += 1
            else: # Unanswerable question
                processed_unanswerable += 1
                
                # Considered correct if retriever found *no* results
                if not found_any:
                    unanswerable_correctly_rejected += 1
                else:
                    unanswerable_false_positives += 1
                    logging.info(f"False positive: {retriever_name} returned results for unanswerable question ID: {result['question_id']} - '{result['question'][:50]}...'")

        # Calculate final metrics
        if processed_answerable > 0:
            metrics[retriever_name]['MRR'] = sum_reciprocal_rank / processed_answerable
            for k in eval_k:
                metrics[retriever_name][f'Recall@{k}'] = correct_hits_at_k[k] / processed_answerable
        else:
             metrics[retriever_name]['MRR'] = 0.0
             for k in eval_k:
                 metrics[retriever_name][f'Recall@{k}'] = 0.0

        if processed_unanswerable > 0:
            metrics[retriever_name]['Unanswerable_Accuracy'] = unanswerable_correctly_rejected / processed_unanswerable
            metrics[retriever_name]['False_Positive_Rate'] = unanswerable_false_positives / processed_unanswerable
        else:
             # If no unanswerable questions, accuracy is undefined or perfect? Set to 1.0 or NaN.
             metrics[retriever_name]['Unanswerable_Accuracy'] = 1.0
             metrics[retriever_name]['False_Positive_Rate'] = 0.0
             
        # Add average latency metric (in milliseconds for better readability)
        if latency_count > 0:
            metrics[retriever_name]['Avg_Latency_ms'] = (total_latency / latency_count) * 1000
        else:
            metrics[retriever_name]['Avg_Latency_ms'] = np.nan

    # Format into DataFrame
    metrics_df = pd.DataFrame(metrics).T # Transpose for better readability
    # Define desired column order
    metric_order = ['MRR'] + [f'Recall@{k}' for k in sorted(eval_k)] + ['Unanswerable_Accuracy', 'False_Positive_Rate', 'Avg_Latency_ms']
    # Ensure all columns exist, add NaN if not (e.g., if no unanswerable questions)
    for col in metric_order:
         if col not in metrics_df.columns:
              metrics_df[col] = np.nan
    metrics_df = metrics_df[metric_order] # Reorder columns

    return metrics_df