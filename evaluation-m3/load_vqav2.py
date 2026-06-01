"""
Dataset loading and inspection script for VQAv2.

Loads the validation split of lmms-lab/VQAv2 via HuggingFace datasets
and outputs dataset statistics and structure.
"""

from __future__ import annotations

import argparse
from datasets import load_dataset


def inspect_dataset(limit: int | None = 5) -> None:
    """Load and inspect the VQAv2 dataset."""
    print("Loading lmms-lab/VQAv2 dataset (all splits combined)...")
    # Load all splits concatenated
    dataset = load_dataset("lmms-lab/VQAv2", split="all")
    
    print("\n=== Dataset Statistics ===")
    print(f"Total samples: {len(dataset):,}")
    
    print("\n=== Dataset Features ===")
    for name, feature in dataset.features.items():
        print(f" - {name}: {feature}")
        
    print(f"\n=== Sample Inspection (First {limit}) ===")
    for i in range(min(limit or 0, len(dataset))):
        sample = dataset[i]
        print(f"\nSample #{i+1}:")
        print(f"  Question ID:      {sample['question_id']}")
        print(f"  Image ID:         {sample['image_id']}")
        print(f"  Question:         '{sample['question']}'")
        print(f"  Question Type:    '{sample.get('question_type', 'N/A')}'")
        print(f"  Answer Type:      '{sample.get('answer_type', 'N/A')}'")
        print(f"  MC Answer:        '{sample.get('multiple_choice_answer', 'N/A')}'")
        
        # Format the list of answers for readability
        answers = sample.get('answers', [])
        ans_list = [ans['answer'] for ans in answers] if answers else []
        print(f"  Annotator Answers: {ans_list}")
        
        # Verify image exists and print dimensions
        image = sample['image']
        print(f"  Image Spec:       {image.mode} | {image.size[0]}x{image.size[1]} (WxH)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect the VQAv2 dataset.")
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of samples to inspect and display."
    )
    args = parser.parse_args()
    inspect_dataset(args.limit)
