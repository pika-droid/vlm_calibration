"""
Multi-scale evaluation harness for VLM Calibration.

Sweeps across token depths for each VQAv2 sample, collects generated answers,
computes confidence statistics, and measures prediction stability using
sentence embeddings. Implements robust checkpointing for long runs.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re
import time
import numpy as np
import torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from evaluation.config import EvalConfig, DEFAULT_CONFIG
from evaluation.model_wrapper import MQTLLaVAWrapper
from visualization.variance_plots import generate_variance_plots
from visualization.reliability_diagram import plot_reliability_diagrams
from visualization.ece_summary import generate_ece_summary


def compute_vqa_accuracy(pred_answer: str, gt_answers: list[str]) -> float:
    """Compute standard VQA accuracy for a predicted answer.

    Formula: Acc = min(1.0, count(pred_answer) / 3)
    Normalizes both predictions and ground truth answers before matching.
    """
    def normalize(text: str) -> str:
        # Standard light normalization for matching
        text = text.lower().strip()
        # Remove punctuation
        text = re.sub(r'[^\w\s]', '', text)
        # Remove articles
        text = re.sub(r'\b(a|an|the)\b', '', text)
        return ' '.join(text.split())

    pred_norm = normalize(pred_answer)
    match_count = 0
    for gt in gt_answers:
        if normalize(gt) == pred_norm:
            match_count += 1
            
    return min(1.0, match_count / 3.0)


def run_evaluation(config: EvalConfig) -> None:
    """Run the multi-scale evaluation loop."""
    # Ensure directories exist
    config.ensure_dirs()
    
    print("==========================================")
    print(" Starting VLM Calibration Evaluation Harness")
    print("==========================================")
    
    # 1. Load VQAv2 dataset
    print(f"Loading {config.dataset_name} ({config.dataset_split} split)...")
    dataset = load_dataset(config.dataset_name, split=config.dataset_split)
    
    # Handle subset slicing
    total_samples = len(dataset)
    if config.subset_size is not None:
        dataset = dataset.select(range(min(config.subset_size, total_samples)))
        print(f"Using a subset of {len(dataset):,} samples (out of {total_samples:,}).")
    else:
        print(f"Running on all {len(dataset):,} samples.")
        
    # 2. Initialize Model Wrapper
    print(f"Loading VLM from {config.model_path} in {config.precision}...")
    vlm = MQTLLaVAWrapper(
        model_path=config.model_path,
        precision=config.precision
    )
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # 3. Initialize Sentence Embeddings Model (runs on CPU to conserve GPU VRAM)
    print(f"Loading sentence similarity model '{config.embedding_model}'...")
    embed_model = SentenceTransformer(config.embedding_model, device="cpu")
    
    # 4. Check for existing checkpoints to resume progress
    latest_checkpoint = None
    checkpoint_file = config.results_jsonl_path()
    start_idx = 0
    
    if checkpoint_file.exists():
        # Count completed samples in the JSONL file
        try:
            with open(checkpoint_file, "r") as f:
                completed = sum(1 for _ in f)
            if completed < len(dataset):
                start_idx = completed
                print(f"Resuming from checkpoint. Already completed {start_idx:,} samples.")
            else:
                print("All samples are already processed. Exiting.")
                return
        except Exception as e:
            print(f"Error reading checkpoint: {e}. Starting from scratch.")
            
    print(f"Output JSONL path: {config.results_jsonl_path()}")
    print(f"Summary CSV path:  {config.summary_csv_path()}")
    
    # Open JSONL in append mode ('a')
    with open(config.results_jsonl_path(), "a" if start_idx > 0 else "w", encoding="utf-8") as out_file:
        
        # 5. Core Evaluation Loop
        loop_range = range(start_idx, len(dataset))
        progress_bar = tqdm(loop_range, desc="Evaluating", unit="sample")
        
        for idx in progress_bar:
            sample = dataset[idx]
            question_id = sample["question_id"]
            image_id = sample["image_id"]
            question = sample["question"]
            question_type = sample["question_type"]
            answer_type = sample["answer_type"]
            
            # Ground truth list
            answers_field = sample.get("answers", [])
            gt_list = [ans["answer"] for ans in answers_field] if answers_field else []
            has_gt = len(gt_list) > 0
            
            # Run sweep across token counts for this sample
            results_by_m = {}
            answers = []
            logprobs = []
            softmax_confs = []
            vqa_accuracies = []
            
            try:
                sweep_results = vlm.sweep_optimized(
                    image=sample["image"],
                    question=question,
                    token_counts=config.token_sweep,
                    max_new_tokens=config.max_new_tokens,
                    temperature=config.temperature
                )
                
                for m in config.token_sweep:
                    res = sweep_results[m]
                    vqa_acc = compute_vqa_accuracy(res["answer"], gt_list) if has_gt else None
                    
                    results_by_m[str(m)] = {
                        "answer": res["answer"],
                        "log_prob": res["log_prob"],
                        "avg_log_prob": res["avg_log_prob"],
                        "softmax_conf": res["softmax_conf"],
                        "avg_softmax_conf": res["avg_softmax_conf"],
                        "vqa_accuracy": vqa_acc
                    }
                    answers.append(res["answer"])
                    logprobs.append(res["avg_log_prob"])
                    softmax_confs.append(res["avg_softmax_conf"])
                    if has_gt:
                        vqa_accuracies.append(vqa_acc)
                    
                # Compute answer stability using sentence embeddings
                # Normalizing embeddings makes cosine similarity a simple dot product
                embeddings = embed_model.encode(answers, show_progress_bar=False)
                norm_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
                similarity_matrix = np.dot(norm_embeddings, norm_embeddings.T)
                
                # Mean pairwise cosine similarity (excluding self-similarity)
                n_metrics = len(config.token_sweep)
                triu_indices = np.triu_indices(n_metrics, k=1)
                answer_stability = float(np.mean(similarity_matrix[triu_indices]))
                
                # Cosine similarity to the final fine-grained answer (m=256)
                similarity_to_final = similarity_matrix[:, -1].tolist()
                mean_sim_to_final = float(np.mean(similarity_to_final[:-1])) # exclude final to final
                
                # Aggregate statistics
                stats = {
                    "answer_stability": answer_stability,
                    "mean_similarity_to_final": mean_sim_to_final,
                    "confidence_mean_logprob": float(np.mean(logprobs)),
                    "confidence_var_logprob": float(np.var(logprobs)),
                    "confidence_mean_softmax": float(np.mean(softmax_confs)),
                    "confidence_var_softmax": float(np.var(softmax_confs)),
                    "accuracy_mean": float(np.mean(vqa_accuracies)) if has_gt else 0.0,
                    "first_correct_m": int(config.token_sweep[vqa_accuracies.index(max(vqa_accuracies))]) if has_gt and max(vqa_accuracies) > 0.0 else -1,
                    "num_unique_answers": int(len(set(answers)))
                }
                
                # Compile entry
                entry = {
                    "question_id": question_id,
                    "image_id": image_id,
                    "question": question,
                    "question_type": question_type,
                    "answer_type": answer_type,
                    "gt_answers": gt_list,
                    "results_by_m": results_by_m,
                    "statistics": stats
                }
                
                # Write to JSONL
                out_file.write(json.dumps(entry, ensure_ascii=False) + "\n")
                out_file.flush()
                
                # Update progress bar
                progress_bar.set_postfix({
                    "stability": f"{answer_stability:.2f}",
                    "avg_acc": f"{stats['accuracy_mean']:.2f}"
                })

                # Trigger intermediate updates and archiving
                processed_count = idx + 1
                if processed_count % config.checkpoint_interval == 0:
                    print(f"\n--- [Checkpoint {processed_count}] Updating intermediate visualizations... ---")
                    try:
                        # Compile current CSV summary first
                        compile_summary_csv(config)
                        
                        generate_variance_plots(config)
                        plot_reliability_diagrams(config)
                        generate_ece_summary(config)
                    except Exception as ve:
                        print(f"Warning: Intermediate visualization update failed: {ve}")
                        
                if processed_count % config.archive_interval == 0:
                    print(f"\n--- [Archive Snapshot {processed_count}] Archiving plots and statistics... ---")
                    try:
                        # Compile current CSV snapshot
                        compile_summary_csv(config)
                        
                        # Create unique snapshot directory
                        snapshot_dir = Path(config.plots_dir) / f"snapshot_{processed_count}"
                        snapshot_dir.mkdir(parents=True, exist_ok=True)
                        
                        # Configure a snapshot copy of config pointing to snapshot_dir
                        snapshot_config = EvalConfig(
                            model_path=config.model_path,
                            subset_size=config.subset_size,
                            output_dir=config.output_dir,
                            plots_dir=str(snapshot_dir),
                            archive_interval=config.archive_interval,
                            checkpoint_interval=config.checkpoint_interval,
                            num_workers=config.num_workers
                        )
                        # Run the plots with the snapshot config
                        generate_variance_plots(snapshot_config)
                        plot_reliability_diagrams(snapshot_config)
                        generate_ece_summary(snapshot_config)
                        
                        # Copy the current summary CSV snapshot into the snapshot directory
                        import shutil
                        if config.summary_csv_path().exists():
                            shutil.copy(config.summary_csv_path(), snapshot_dir / f"summary_statistics_{processed_count}.csv")
                            
                        print(f"Snapshot successfully archived at: {snapshot_dir}")
                    except Exception as ae:
                        print(f"Warning: Archiving snapshot failed: {ae}")
                
            except Exception as e:
                print(f"\nError processing sample {question_id} at index {idx}: {e}")
                continue
                
    # 6. Post-processing: Compile summary CSV
    print("\nEvaluation completed. Generating summary CSV statistics...")
    try:
        compile_summary_csv(config)
    except Exception as e:
        print(f"Error compiling summary CSV: {e}")
        
    print("\n==========================================")
    print(" Evaluation Harness Finished successfully!")
    print("==========================================")


def compile_summary_csv(config: EvalConfig) -> None:
    """Read the JSONL output and generate a flat summary CSV of all statistics."""
    jsonl_path = config.results_jsonl_path()
    csv_path = config.summary_csv_path()
    
    if not jsonl_path.exists():
        print(f"Error: JSONL file not found at {jsonl_path}")
        return
        
    headers = [
        "question_id", "image_id", "question_type", "answer_type",
        "answer_stability", "mean_similarity_to_final",
        "confidence_mean_logprob", "confidence_var_logprob",
        "confidence_mean_softmax", "confidence_var_softmax",
        "accuracy_mean", "num_unique_answers"
    ]
    
    # Add per-m accuracies and confidences
    for m in config.token_sweep:
        headers.extend([
            f"acc_m_{m}",
            f"conf_logprob_m_{m}",
            f"conf_softmax_m_{m}"
        ])
        
    with open(jsonl_path, "r", encoding="utf-8") as f_in, \
         open(csv_path, "w", newline="", encoding="utf-8") as f_out:
        
        writer = csv.writer(f_out)
        writer.writerow(headers)
        
        for line in f_in:
            data = json.loads(line)
            stats = data["statistics"]
            
            row = [
                data["question_id"],
                data["image_id"],
                data["question_type"],
                data["answer_type"],
                stats["answer_stability"],
                stats["mean_similarity_to_final"],
                stats["confidence_mean_logprob"],
                stats["confidence_var_logprob"],
                stats["confidence_mean_softmax"],
                stats["confidence_var_softmax"],
                stats["accuracy_mean"],
                stats["num_unique_answers"]
            ]
            
            # Extract per-m values
            for m in config.token_sweep:
                m_data = data["results_by_m"][str(m)]
                row.extend([
                    m_data["vqa_accuracy"],
                    m_data["avg_log_prob"],
                    m_data["avg_softmax_conf"]
                ])
                
            writer.writerow(row)
            
    print(f"Summary statistics exported to {csv_path}")


if __name__ == "__main__":
    # Argument parser
    parser = argparse.ArgumentParser(description="VLM Calibration Sweep Harness.")
    parser.add_argument("--model-path", type=str, default=DEFAULT_CONFIG.model_path)
    parser.add_argument("--subset-size", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_CONFIG.output_dir)
    args = parser.parse_args()
    
    # Load configuration
    run_config = EvalConfig(
        model_path=args.model_path,
        subset_size=args.subset_size,
        output_dir=args.output_dir
    )
    
    run_evaluation(run_config)
