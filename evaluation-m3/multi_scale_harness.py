"""
Multi-scale evaluation harness for VLM Calibration using M3-LLaVA.

Sweeps across token depths for each VQAv2 sample, collects generated answers,
computes confidence statistics, and measures prediction stability using
sentence embeddings. Implements robust checkpointing for long runs.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
import re
import random
import shutil
import sys
import time
import numpy as np
import torch
from datasets import load_dataset
from dataclasses import replace
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from .config import EvalConfig, DEFAULT_CONFIG
from .model_wrapper import M3LLaVAWrapper
from .visualization.variance_plots import generate_variance_plots
from .visualization.reliability_diagram import plot_reliability_diagrams
from .visualization.ece_summary import generate_ece_summary


def set_seed(seed: int = 42) -> None:
    """Set all random seeds for full reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logging(config: EvalConfig) -> logging.Logger:
    """Set up file and console logger."""
    config.ensure_dirs()
    log_file = Path(config.logs_dir) / "evaluation.log"
    
    logger = logging.getLogger("VLM_M3_Calibration")
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        file_handler.setFormatter(logging.Formatter(
            fmt="[%(asctime)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        ))
        file_handler.setLevel(logging.INFO)
        logger.addHandler(file_handler)
        
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(logging.Formatter(fmt="%(message)s"))
        console_handler.setLevel(logging.INFO)
        logger.addHandler(console_handler)
        
    return logger


def compute_vqa_accuracy(pred_answer: str, gt_answers: list[str]) -> float:
    """Compute standard VQA accuracy for a predicted answer."""
    def normalize(text: str) -> str:
        text = text.lower().strip()
        text = re.sub(r'[^\w\s]', '', text)
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
    config.ensure_dirs()
    logger = setup_logging(config)
    set_seed(config.seed)
    
    logger.info("==========================================")
    logger.info(" Starting M3 VLM Calibration Evaluation Harness")
    logger.info("==========================================")
    
    # 1. Load VQAv2 dataset
    logger.info(f"Loading {config.dataset_name} ({config.dataset_split} split)...")
    dataset = load_dataset(config.dataset_name, split=config.dataset_split)
    
    total_samples = len(dataset)
    if config.subset_size is not None:
        dataset = dataset.select(range(min(config.subset_size, total_samples)))
        logger.info(f"Using a subset of {len(dataset):,} samples (out of {total_samples:,}).")
    else:
        logger.info(f"Running on all {len(dataset):,} samples.")
        
    # 2. Initialize Model Wrapper
    logger.info(f"Loading VLM from {config.model_path} in {config.precision}...")
    vlm = M3LLaVAWrapper(
        model_path=config.model_path,
        precision=config.precision
    )
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # 3. Initialize Sentence Embeddings Model (CPU to conserve GPU VRAM)
    logger.info(f"Loading sentence similarity model '{config.embedding_model}'...")
    embed_model = SentenceTransformer(config.embedding_model, device="cpu")
    
    # 4. Check for existing checkpoints to resume progress
    checkpoint_file = config.results_jsonl_path()
    start_idx = 0
    
    if checkpoint_file.exists():
        try:
            with open(checkpoint_file, "r", encoding="utf-8") as f:
                completed = sum(1 for _ in f)
            if completed < len(dataset):
                start_idx = completed
                logger.info(f"Resuming from checkpoint. Already completed {start_idx:,} samples.")
            else:
                logger.info("All samples are already processed. Exiting.")
                return
        except Exception as e:
            logger.info(f"Error reading checkpoint: {e}. Starting from scratch.")
            
    logger.info(f"Output JSONL path: {config.results_jsonl_path()}")
    logger.info(f"Summary CSV path:  {config.summary_csv_path()}")
    
    with open(config.results_jsonl_path(), "a" if start_idx > 0 else "w", encoding="utf-8") as out_file:
        loop_range = range(start_idx, len(dataset))
        progress_bar = tqdm(loop_range, desc="Evaluating", unit="sample")
        
        for idx in progress_bar:
            sample = dataset[idx]
            question_id = sample["question_id"]
            image_id = sample["image_id"]
            question = sample["question"]
            question_type = sample["question_type"]
            answer_type = sample["answer_type"]
            
            answers_field = sample.get("answers", [])
            gt_list = [ans["answer"] for ans in answers_field] if answers_field else []
            has_gt = len(gt_list) > 0
            
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
                embeddings = embed_model.encode(answers, show_progress_bar=False)
                norm_embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10)
                similarity_matrix = np.dot(norm_embeddings, norm_embeddings.T)
                
                n_metrics = len(config.token_sweep)
                triu_indices = np.triu_indices(n_metrics, k=1)
                answer_stability = float(np.mean(similarity_matrix[triu_indices]))
                
                similarity_to_final = similarity_matrix[:, -1].tolist()
                mean_sim_to_final = float(np.mean(similarity_to_final[:-1]))
                
                stats = {
                    "answer_stability": answer_stability,
                    "mean_similarity_to_final": mean_sim_to_final,
                    "confidence_mean_logprob": float(np.mean(logprobs)),
                    "confidence_var_logprob": float(np.var(logprobs)),
                    "confidence_mean_softmax": float(np.mean(softmax_confs)),
                    "confidence_var_softmax": float(np.var(softmax_confs)),
                    "accuracy_mean": float(np.mean(vqa_accuracies)) if has_gt else 0.0,
                    "first_correct_m": int(next((config.token_sweep[i] for i, a in enumerate(vqa_accuracies) if a > 0.0), -1)) if has_gt else -1,
                    "num_unique_answers": int(len(set(answers)))
                }
                
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
                
                out_file.write(json.dumps(entry, ensure_ascii=False) + "\n")
                out_file.flush()
                
                progress_bar.set_postfix({
                    "stability": f"{answer_stability:.2f}",
                    "avg_acc": f"{stats['accuracy_mean']:.2f}"
                })

                processed_count = idx + 1
                if processed_count % config.checkpoint_interval == 0:
                    logger.info(f"\n--- [Checkpoint {processed_count}] Updating intermediate visualizations... ---")
                    try:
                        compile_summary_csv(config)
                        generate_variance_plots(config)
                        plot_reliability_diagrams(config)
                        generate_ece_summary(config)
                    except Exception as ve:
                        logger.info(f"Warning: Intermediate visualization update failed: {ve}")
                        
                if processed_count % config.archive_interval == 0:
                    logger.info(f"\n--- [Archive Snapshot {processed_count}] Archiving plots and statistics... ---")
                    try:
                        compile_summary_csv(config)
                        
                        snapshot_dir = Path(config.plots_dir) / f"snapshot_{processed_count}"
                        snapshot_dir.mkdir(parents=True, exist_ok=True)
                        
                        snapshot_config = replace(config, plots_dir=str(snapshot_dir))
                        generate_variance_plots(snapshot_config)
                        plot_reliability_diagrams(snapshot_config)
                        generate_ece_summary(snapshot_config)
                        
                        if config.summary_csv_path().exists():
                            shutil.copy(config.summary_csv_path(), snapshot_dir / f"summary_statistics_{processed_count}.csv")
                            
                        logger.info(f"Snapshot successfully archived at: {snapshot_dir}")
                    except Exception as ae:
                        logger.info(f"Warning: Archiving snapshot failed: {ae}")
                
            except Exception as e:
                logger.info(f"\nError processing sample {question_id} at index {idx}: {e}")
                continue
                
    logger.info("\nEvaluation completed. Generating summary CSV statistics...")
    try:
        compile_summary_csv(config)
    except Exception as e:
        logger.info(f"Error compiling summary CSV: {e}")
        
    logger.info("\n==========================================")
    logger.info(" Evaluation Harness Finished successfully!")
    logger.info("==========================================")


def compile_summary_csv(config: EvalConfig) -> None:
    """Read the JSONL output and generate a flat summary CSV of all statistics."""
    logger = logging.getLogger("VLM_M3_Calibration")
    jsonl_path = config.results_jsonl_path()
    csv_path = config.summary_csv_path()
    
    if not jsonl_path.exists():
        logger.info(f"Error: JSONL file not found at {jsonl_path}")
        return
        
    headers = [
        "question_id", "image_id", "question_type", "answer_type",
        "answer_stability", "mean_similarity_to_final",
        "confidence_mean_logprob", "confidence_var_logprob",
        "confidence_mean_softmax", "confidence_var_softmax",
        "accuracy_mean", "num_unique_answers"
    ]
    
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
            
            for m in config.token_sweep:
                m_data = data["results_by_m"][str(m)]
                row.extend([
                    m_data["vqa_accuracy"],
                    m_data["avg_log_prob"],
                    m_data["avg_softmax_conf"]
                ])
                
            writer.writerow(row)
            
    logger.info(f"Summary statistics exported to {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="M3 VLM Calibration Sweep Harness.")
    parser.add_argument("--model-path", type=str, default=DEFAULT_CONFIG.model_path)
    parser.add_argument("--subset-size", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_CONFIG.output_dir)
    args = parser.parse_args()
    
    run_config = EvalConfig(
        model_path=args.model_path,
        subset_size=args.subset_size,
        output_dir=args.output_dir
    )
    
    run_evaluation(run_config)
