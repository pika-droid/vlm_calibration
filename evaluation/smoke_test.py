"""
Smoke test to verify that the MQT-LLaVA model loader, prompt preprocessing,
and confidence extraction work together on a few validation samples.
"""

from __future__ import annotations

import torch
from datasets import load_dataset
from evaluation.config import DEFAULT_CONFIG
from evaluation.model_wrapper import MQTLLaVAWrapper


def run_smoke_test() -> None:
    """Run a quick sanity check of the model wrapper on VQAv2 samples."""
    print("==========================================")
    print(" Running VLM Calibration Smoke Test")
    print("==========================================")
    
    # 1. Load VQAv2 dataset
    print(f"Loading {DEFAULT_CONFIG.dataset_name} validation dataset split...")
    dataset = load_dataset(DEFAULT_CONFIG.dataset_name, split=DEFAULT_CONFIG.dataset_split)
    print(f"Loaded {len(dataset):,} samples.")
    
    # Take first sample
    sample = dataset[0]
    question = sample["question"]
    image = sample["image"]
    gt_answers = [ans["answer"] for ans in sample["answers"]]
    
    print("\n--- Sample Info ---")
    print(f"Question ID: {sample['question_id']}")
    print(f"Question:    '{question}'")
    print(f"GT Answers:  {gt_answers}")
    print(f"Image Size:  {image.size} ({image.mode})")
    
    # 2. Load model wrapper
    print(f"\nLoading model '{DEFAULT_CONFIG.model_path}'...")
    wrapper = MQTLLaVAWrapper(
        model_path=DEFAULT_CONFIG.model_path,
        precision=DEFAULT_CONFIG.precision
    )
    
    # 3. Test token count sweep on the sample
    test_token_counts = [16, 256]
    print(f"\nRunning sweep across token counts: {test_token_counts}")
    
    results = wrapper.sweep(
        image=image,
        question=question,
        token_counts=test_token_counts,
        max_new_tokens=DEFAULT_CONFIG.max_new_tokens,
        temperature=DEFAULT_CONFIG.temperature
    )
    
    # 4. Print results
    print("\n=== Generation Results ===")
    for m, res in results.items():
        print(f"\nVisual Token Count (m) = {m}:")
        print(f"  Generated Answer: '{res['answer']}'")
        print(f"  Log-Probability (Sum):   {res['log_prob']:.4f}")
        print(f"  Average Log-Probability: {res['avg_log_prob']:.4f}")
        print(f"  Softmax Confidence:      {res['softmax_conf']:.4f}")
        print(f"  Avg Softmax Confidence:  {res['avg_softmax_conf']:.4f}")
        
        # Verify correctness against ground truth (simple matching)
        is_exact_match = res["answer"].lower() in [ans.lower() for ans in gt_answers]
        print(f"  Is Exact Match:          {is_exact_match}")
        
        # Display token breakdown
        print("  Token Breakdown:")
        for details in res["token_details"]:
            tok_text = repr(details['token_text'])
            print(f"    - Token {tok_text:<12} | Prob: {details['prob']:.4f} | Logprob: {details['logprob']:.4f}")

    print("\n==========================================")
    print(" Smoke test completed successfully!")
    print("==========================================")


if __name__ == "__main__":
    run_smoke_test()
