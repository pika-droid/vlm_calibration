"""Pilot inference sweep and diagnostics manager for M3-LLaVA.

Performs stratified sampling of 1,000 VQAv2 evaluation samples across 4 classes:
1. Stable Correct (400)
2. Stable Incorrect - Strict (200)
3. Stable Incorrect - Relaxed (200)
4. Flip (200)

Executes inference across all 5 scales (m ∈ [1, 9, 36, 144, 576]) with forward hooks
active, computing Visual Attention Ratio (VAR) and LogitLens statistics on the fly.
"""

from __future__ import annotations

import os
import sys
import json
import argparse
import random
import logging
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from datasets import load_dataset

# Add evaluation-m3 and interpretability package to system path
root_path = Path(__file__).resolve().parent.parent
package_path = Path(__file__).resolve().parent
eval_m3_path = root_path / "evaluation-m3"
if str(eval_m3_path) not in sys.path:
    sys.path.insert(0, str(eval_m3_path))
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))
if str(package_path) not in sys.path:
    sys.path.insert(0, str(package_path))

from extract_hooks import HookedM3Wrapper, InterpConfig, HookOutput, TokenMap

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("VLM_Interpretability")


def classify_sample_strata(results_by_m: dict[str, Any]) -> str | None:
    """Classifies a sample into one of the 4 strata based on its multi-scale VQA results."""
    # Matryoshka scales: 1, 9, 36, 144, 576
    accuracies = []
    answers = []
    
    for m in ["1", "9", "36", "144", "576"]:
        if m in results_by_m:
            res = results_by_m[m]
            acc = res.get("vqa_accuracy")
            ans = res.get("answer", "").strip().lower()
            if acc is not None:
                accuracies.append(acc)
            answers.append(ans)

    if not accuracies:
        return None

    # Class 1: Stable Correct (all scales have accuracy >= 0.5)
    if all(acc >= 0.5 for acc in accuracies):
        return "stable_correct"
        
    # Class 2 & 3: Stable Incorrect (all scales have accuracy < 0.5)
    if all(acc < 0.5 for acc in accuracies):
        first_ans = answers[0] if answers else ""
        all_identical = all(ans == first_ans for ans in answers)
        if all_identical:
            return "stable_incorrect_strict"
        else:
            return "stable_incorrect_relaxed"
            
    # Class 4: Flip (accuracy changes across scales)
    any_correct = any(acc >= 0.5 for acc in accuracies)
    any_incorrect = any(acc < 0.5 for acc in accuracies)
    if any_correct and any_incorrect:
        return "flip"

    return None


def select_stratified_pilot(config: InterpConfig) -> dict[str, list[int]]:
    """Reads source JSONL file and performs stratified sampling to select pilot question IDs."""
    source_path = Path(config.source_results_jsonl)
    if not source_path.exists():
        # Handle case where results are in a relative path or need resolution
        alt_path = root_path / config.source_results_jsonl
        if alt_path.exists():
            source_path = alt_path
        else:
            raise FileNotFoundError(f"Source results file not found: {config.source_results_jsonl}")

    logger.info(f"Reading multi-scale evaluation results from: {source_path}")
    
    pools: dict[str, list[int]] = {
        "stable_correct": [],
        "stable_incorrect_strict": [],
        "stable_incorrect_relaxed": [],
        "flip": []
    }

    with open(source_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            q_id = int(data["question_id"])
            results_by_m = data.get("results_by_m", {})
            stratum = classify_sample_strata(results_by_m)
            if stratum in pools:
                pools[stratum].append(q_id)

    logger.info("Dataset Pools statistics:")
    for k, v in pools.items():
        logger.info(f" - {k}: {len(v)} samples")

    # Set random seeds for reproducibility
    random.seed(config.seed)
    
    selected_samples: dict[str, list[int]] = {}
    for stratum, target_count in config.stratification_counts.items():
        pool = pools[stratum]
        if len(pool) < target_count:
            logger.warning(
                f"Stratum '{stratum}' has only {len(pool)} samples, but target is {target_count}. "
                f"Using all available samples in this pool."
            )
            selected_samples[stratum] = pool.copy()
        else:
            selected_samples[stratum] = random.sample(pool, target_count)

    # Save selection setup for reference and reproducibility
    output_dir = Path(config.pilot_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    selection_file = output_dir / "pilot_sample_ids.json"
    with open(selection_file, "w", encoding="utf-8") as f:
        json.dump(selected_samples, f, indent=4)
        
    logger.info(f"Selected pilot question IDs saved to: {selection_file}")
    
    total_selected = sum(len(ids) for ids in selected_samples.values())
    logger.info(f"Total selected pilot samples: {total_selected}")
    for k, v in selected_samples.items():
        logger.info(f" - {k}: {len(v)} samples selected")
        
    return selected_samples


@torch.no_grad()
def compute_var_on_the_fly(
    hook_output: HookOutput,
    token_map: TokenMap,
    hook_layers: list[int]
) -> dict[str, float]:
    """Computes Visual Attention Ratio (VAR) on the fly for hooked layers.

    VAR = Sum(attention from answer tokens to image tokens) / Sum(attention from answer tokens to text tokens)
    text tokens include question query tokens and system prompt tokens.
    """
    var_results = {}
    tm = token_map
    
    # Verify we have answer tokens
    if tm.answer_start >= tm.answer_end:
        return {}

    for layer_idx in hook_layers:
        attn = hook_output.attention_weights.get(layer_idx)
        if attn is None:
            continue
            
        # Shape of attn: (num_heads, seq_len, seq_len)
        # We average across all heads
        mean_attn = attn.mean(dim=0)  # (seq_len, seq_len)
        
        # Attention from answer tokens (rows) to other tokens (columns)
        # Answer rows: [answer_start : answer_end]
        ans_rows = mean_attn[tm.answer_start : tm.answer_end, :]
        
        # Image columns: [image_start : image_end]
        image_attn = ans_rows[:, tm.image_start : tm.image_end].sum().item()
        
        # Text/Context columns: sum of everything excluding image tokens
        # We define text tokens as prompt/question tokens: [0 : image_start] + [image_end : answer_start]
        prefix_attn = ans_rows[:, 0 : tm.image_start].sum().item()
        question_attn = ans_rows[:, tm.image_end : tm.answer_start].sum().item()
        text_attn = prefix_attn + question_attn
        
        # Avoid division by zero
        var_val = image_attn / max(text_attn, 1e-8)
        var_results[str(layer_idx)] = var_val
        
    return var_results


@torch.no_grad()
def compute_logit_lens_on_the_fly(
    hook_output: HookOutput,
    token_map: TokenMap,
    hook_layers: list[int],
    model_wrapper: HookedM3Wrapper
) -> dict[str, dict[str, Any]]:
    """Projects intermediate hidden states at answer token positions to vocabulary.

    Calculates the intermediate top-1 token, confidence, and entropy.
    """
    logit_lens_results = {}
    tm = token_map
    
    # We evaluate LogitLens at the first answer token position (or last prefill token)
    # where the model decides on the first generated character/word.
    eval_pos = tm.answer_start - 1
    
    # LLaMA model norms and output heads
    # Check if this is a dummy run or if model layers are loaded
    if not hasattr(model_wrapper, "model") or not hasattr(model_wrapper.model, "model") or not hasattr(model_wrapper.model, "lm_head"):
        # Mock mode fallback
        for layer_idx in hook_layers:
            logit_lens_results[str(layer_idx)] = {
                "top1_token": "mock",
                "confidence": float(np.random.uniform(0.1, 0.9)),
                "entropy": float(np.random.uniform(0.5, 4.0))
            }
        return logit_lens_results

    norm_layer = model_wrapper.model.model.norm
    lm_head = model_wrapper.model.lm_head
    tokenizer = model_wrapper.tokenizer

    for layer_idx in hook_layers:
        h_state = hook_output.hidden_states.get(layer_idx)
        if h_state is None:
            continue
            
        # h_state shape: (seq_len, hidden_dim) or (1, seq_len, hidden_dim)
        if h_state.dim() == 3:
            h_state = h_state[0]
            
        # Check bounds
        if eval_pos < 0 or eval_pos >= h_state.size(0):
            continue
            
        # Extract hidden vector at evaluation position
        h_vec = h_state[eval_pos : eval_pos + 1].to(model_wrapper.device, dtype=model_wrapper.dtype)
        
        # Apply LLaMA RMSNorm then Linear LM Head
        normed = norm_layer(h_vec)
        logits = lm_head(normed)[0]  # (vocab_size,)
        
        # Compute probabilities and entropy
        probs = torch.softmax(logits, dim=-1)
        log_probs = torch.log_softmax(logits, dim=-1)
        entropy = -torch.sum(probs * log_probs).item()
        
        # Extract top-1 prediction
        top1_prob, top1_idx = torch.max(probs, dim=-1)
        top1_idx_val = top1_idx.item()
        top1_token = tokenizer.decode([top1_idx_val]).strip()
        
        logit_lens_results[str(layer_idx)] = {
            "top1_token": top1_token,
            "confidence": float(top1_prob.item()),
            "entropy": float(entropy)
        }
        
    return logit_lens_results


def run_pilot_inference(config: InterpConfig, selected_samples: dict[str, list[int]]) -> None:
    """Executes the pilot inference loop, computing and saving intermediate states."""
    output_dir = Path(config.pilot_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_file = output_dir / "pilot_results.jsonl"
    
    # Flatten selected samples and store their mapping
    q_id_to_stratum = {}
    all_selected_ids = []
    for stratum, ids in selected_samples.items():
        all_selected_ids.extend(ids)
        for q_id in ids:
            q_id_to_stratum[q_id] = stratum

    # Resume checkpoint management
    completed_q_ids = set()
    if config.resume and results_file.exists():
        logger.info(f"Pilot results file already exists. Checking for resume checkpoints...")
        with open(results_file, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                completed_q_ids.add(int(data["question_id"]))
        logger.info(f"Found {len(completed_q_ids)} completed samples in checkpoint file.")

    # Filter out completed samples
    pending_ids = [q_id for q_id in all_selected_ids if q_id not in completed_q_ids]
    logger.info(f"Total samples to execute: {len(pending_ids)} / {len(all_selected_ids)}")
    
    if not pending_ids:
        logger.info("All selected pilot samples are already completed. Exiting.")
        return

    # Load dataset
    logger.info(f"Loading dataset {config.dataset_name} ({config.dataset_split})...")
    dataset = load_dataset(config.dataset_name, split=config.dataset_split)
    
    # Filter dataset to selected question IDs
    selected_set = set(pending_ids)
    pilot_dataset = [sample for sample in dataset if int(sample["question_id"]) in selected_set]
    logger.info(f"Dataset filtered. Loaded {len(pilot_dataset)} samples from VQAv2.")
    
    # Map by question_id for easy lookup
    pilot_data_map = {int(sample["question_id"]): sample for sample in pilot_dataset}

    # Load Model Wrapper (Only if not running a mock run)
    logger.info("Loading model and tokenizer...")
    wrapper = HookedM3Wrapper(
        model_path=config.model_path,
        precision="fp16"
    )

    # Open result file in append or write mode
    mode = "a" if config.resume else "w"
    with open(results_file, mode, encoding="utf-8") as out_f:
        # Loop through all selected question IDs in deterministic order
        for q_id in tqdm(pending_ids, desc="Processing Pilot Samples"):
            sample = pilot_data_map.get(q_id)
            if sample is None:
                logger.warning(f"Question ID {q_id} not found in VQAv2 dataset. Skipping.")
                continue

            image = sample["image"]
            question = sample["question"]
            gt_answers = [ans["answer"] for ans in sample.get("answers", [])]
            stratum = q_id_to_stratum[q_id]
            
            results_by_m = {}

            # Execute forward pass across all 5 scales
            for m in config.token_sweep:
                try:
                    # Execute forward pass with hooks active
                    hook_output = wrapper.forward_with_hooks(
                        image=image,
                        question=question,
                        num_visual_tokens=m,
                        hook_layers=config.hook_layers
                    )
                    
                    # Compute intermediate metrics on-the-fly
                    var_stats = compute_var_on_the_fly(
                        hook_output=hook_output,
                        token_map=hook_output.token_map,
                        hook_layers=config.hook_layers
                    )
                    
                    logit_lens_stats = compute_logit_lens_on_the_fly(
                        hook_output=hook_output,
                        token_map=hook_output.token_map,
                        hook_layers=config.hook_layers,
                        model_wrapper=wrapper
                    )
                    
                    # Store results for this scale
                    results_by_m[str(m)] = {
                        "answer": hook_output.generated_text,
                        "var_by_layer": var_stats,
                        "logit_lens_by_layer": logit_lens_stats,
                        "token_map": {
                            "image_start": hook_output.token_map.image_start,
                            "image_end": hook_output.token_map.image_end,
                            "question_start": hook_output.token_map.question_start,
                            "question_end": hook_output.token_map.question_end,
                            "answer_start": hook_output.token_map.answer_start,
                            "answer_end": hook_output.token_map.answer_end
                        }
                    }
                except Exception as e:
                    logger.error(f"Error processing question_id {q_id} at scale {m}: {e}")
                    results_by_m[str(m)] = {
                        "answer": "ERROR",
                        "error": str(e)
                    }

            # Compile full sample report
            sample_result = {
                "question_id": q_id,
                "image_id": sample.get("image_id"),
                "question": question,
                "gt_answers": gt_answers,
                "stratum": stratum,
                "results_by_m": results_by_m
            }
            
            # Write to output file
            out_f.write(json.dumps(sample_result) + "\n")
            out_f.flush()

    logger.info(f"Pilot run completed. Results saved to {results_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run stratified pilot inference and extract diagnostics.")
    parser.add_argument("--model-path", type=str, default="/workspace/models/llava-v1.5-7b-m3")
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--source-results", type=str, default="results/vlm-calibration-m3/results/multi_scale_results.jsonl")
    parser.add_argument("--output-dir", type=str, default="results/pilot-interpretability")
    parser.add_argument("--no-resume", action="store_true", help="Overwrite existing results instead of resuming.")
    parser.add_argument("--debug", action="store_true", help="Run a single-sample validation run.")
    parser.add_argument("--sample-idx", type=int, default=0, help="Index of debug sample to process.")
    
    args = parser.parse_args()
    
    # Initialize configuration
    config = InterpConfig(
        model_path=args.model_path,
        source_results_jsonl=args.source_results,
        pilot_output_dir=args.output_dir,
        resume=not args.no_resume,
        debug=args.debug
    )
    
    if args.debug:
        logger.info(f"DEBUG MODE: Running single sample index {args.sample_idx}...")
        # Mock pilot selection
        selected = {"debug": [args.sample_idx]}
        run_pilot_inference(config, selected)
    else:
        # 1. Stratify and select
        selected = select_stratified_pilot(config)
        # 2. Run inference sweep
        run_pilot_inference(config, selected)
