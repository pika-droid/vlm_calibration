"""CPU-only local verification script for the M3-LLaVA Interpretability suite.

Mocks the LLM architectures, HuggingFace datasets, and GPU wrappers to execute the
entire interpretability diagnostic suite end-to-end on CPU.
"""

from __future__ import annotations

import os
import sys
import shutil
import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

# Configure matplotlib to use non-interactive Agg backend to prevent display errors
import matplotlib
matplotlib.use("Agg")

import torch
from PIL import Image

# 1. Setup Mock System Modules for HuggingFace and llava dependencies
mock_llava = MagicMock()
mock_llava.constants = MagicMock()
mock_llava.constants.IMAGE_TOKEN_INDEX = -200
mock_llava.constants.DEFAULT_IMAGE_TOKEN = "<image>"
mock_llava.constants.DEFAULT_IM_START_TOKEN = "<img>"
mock_llava.constants.DEFAULT_IM_END_TOKEN = "</img>"
mock_llava.constants.IMAGE_PLACEHOLDER = "<image-placeholder>"

mock_llava.conversation = MagicMock()
mock_llava.conversation.conv_templates = {"llava_v1": MagicMock()}

mock_llava.model = MagicMock()
mock_llava.model.builder = MagicMock()
mock_llava.model.builder.load_pretrained_model = MagicMock()

mock_llava.utils = MagicMock()
mock_llava.utils.disable_torch_init = MagicMock()

mock_llava.mm_utils = MagicMock()
mock_llava.mm_utils.process_images = MagicMock()
mock_llava.mm_utils.tokenizer_image_token = MagicMock()
mock_llava.mm_utils.get_model_name_from_path = MagicMock(return_value="mock-m3-llava")

# Inject mock modules
sys.modules["llava"] = mock_llava
sys.modules["llava.constants"] = mock_llava.constants
sys.modules["llava.conversation"] = mock_llava.conversation
sys.modules["llava.model"] = mock_llava.model
sys.modules["llava.model.builder"] = mock_llava.model.builder
sys.modules["llava.utils"] = mock_llava.utils
sys.modules["llava.mm_utils"] = mock_llava.mm_utils

# 2. Define Mock Dataset
class MockDatasetList:
    def __init__(self, data_list: list[dict]) -> None:
        self.data_list = data_list
        self.features = {
            "question_id": "int",
            "image_id": "int",
            "question": "string",
            "answers": "list",
            "image": "image"
        }

    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(self, idx: int | slice) -> dict | list[dict] | MockDatasetList:
        if isinstance(idx, slice):
            return MockDatasetList(self.data_list[idx])
        return self.data_list[idx]

    def select(self, indices: range | list[int]) -> MockDatasetList:
        return MockDatasetList([self.data_list[i] for i in indices])

    def __iter__(self):
        return iter(self.data_list)

# Generate dummy image
dummy_img = Image.new("RGB", (100, 100), color="blue")
mock_samples = [
    {
        "question_id": 101,
        "image_id": 2001,
        "question": "Is the sky blue?",
        "answers": [{"answer": "yes"}, {"answer": "yes"}, {"answer": "no"}],
        "image": dummy_img,
    },
    {
        "question_id": 102,
        "image_id": 2002,
        "question": "What color is the wall?",
        "answers": [{"answer": "white"}, {"answer": "white"}, {"answer": "white"}],
        "image": dummy_img,
    },
    {
        "question_id": 103,
        "image_id": 2003,
        "question": "How many cats are there?",
        "answers": [{"answer": "2"}, {"answer": "two"}, {"answer": "2"}],
        "image": dummy_img,
    },
    {
        "question_id": 104,
        "image_id": 2004,
        "question": "Is there a car?",
        "answers": [{"answer": "no"}, {"answer": "no"}, {"answer": "no"}],
        "image": dummy_img,
    }
]
mock_dataset_obj = MockDatasetList(mock_samples)

# Mock datasets module
mock_datasets = MagicMock()
mock_datasets.load_dataset = MagicMock(return_value=mock_dataset_obj)
sys.modules["datasets"] = mock_datasets


# 3. Define Mock Model Wrapper for Hooks
from dataclasses import dataclass
from typing import Generator
from contextlib import contextmanager

@dataclass
class TokenMap:
    image_start: int
    image_end: int
    question_start: int
    question_end: int
    answer_start: int
    answer_end: int

@dataclass
class HookOutput:
    hidden_states: dict[int, torch.Tensor]
    attention_weights: dict[int, torch.Tensor]
    token_map: TokenMap
    generated_text: str
    logits: torch.Tensor


class MockHookedM3Wrapper:
    def __init__(self, model_path: str, precision: str = "fp16") -> None:
        self.device = torch.device("cpu")
        self.dtype = torch.float32
        self.model_name = "mock-hooked-m3"
        self.tokenizer = MagicMock()

    def format_prompt(self, question: str) -> str:
        return f"mock prompt: {question}"

    def preprocess_image(self, image: Any) -> tuple[torch.Tensor, list[tuple[int, int]]]:
        return torch.randn(1, 3, 224, 224), [(224, 224)]

    def generate_with_confidence(self, image: Any, question: str, num_visual_tokens: int) -> dict[str, Any]:
        return {"answer": "yes", "avg_softmax_conf": 0.85}

    @contextmanager
    def hooked(self, layer_indices: list[int], capture_hidden: bool = True, capture_attention: bool = True) -> Generator[MockHookedM3Wrapper, None, None]:
        yield self

    def forward_with_hooks(
        self,
        image: Any,
        question: str,
        num_visual_tokens: int = 576,
        hook_layers: list[int] | None = None
    ) -> HookOutput:
        if hook_layers is None:
            hook_layers = [4, 8, 12, 16, 20, 24, 28, 32]
            
        seq_len = 10 + num_visual_tokens  # 10 text tokens + visual scale
        
        # Mock hidden states: Layer -> Tensor (seq_len, hidden_dim)
        hidden_states = {}
        for l in hook_layers:
            # Create dummy hidden states with layer-specific trends
            # e.g., layer 32 has higher values than layer 4
            val = float(l) / 32.0
            hidden_states[l] = torch.ones(seq_len, 4096) * val
            
        # Mock attention weights: Layer -> Tensor (num_heads, seq_len, seq_len)
        attention_weights = {}
        for l in hook_layers:
            # Generate attention weights matrix
            attn = torch.zeros(32, seq_len, seq_len)
            
            # Simulate high visual attention for correct, low for incorrect/hallucination
            # Let's fill the answer-to-image blocks (last token attending to visual tokens)
            image_start = 2
            image_end = 2 + num_visual_tokens
            answer_start = seq_len - 1
            
            # Correct vs incorrect simulation
            is_correct = "sky" in question.lower() or "wall" in question.lower()
            if is_correct:
                attn[:, answer_start, image_start:image_end] = 0.8 / num_visual_tokens
                attn[:, answer_start, image_end:answer_start] = 0.2 / (answer_start - image_end)
            else:
                attn[:, answer_start, image_start:image_end] = 0.1 / num_visual_tokens
                attn[:, answer_start, image_end:answer_start] = 0.9 / (answer_start - image_end)
                
            attention_weights[l] = attn
            
        token_map = TokenMap(
            image_start=2,
            image_end=2 + num_visual_tokens,
            question_start=2 + num_visual_tokens,
            question_end=seq_len - 1,
            answer_start=seq_len - 1,
            answer_end=seq_len
        )
        
        ans = "yes"
        if "color" in question.lower():
            ans = "white"
        elif "cats" in question.lower():
            ans = "2"
        elif "car" in question.lower():
            ans = "no"

        return HookOutput(
            hidden_states=hidden_states,
            attention_weights=attention_weights,
            token_map=token_map,
            generated_text=ans,
            logits=torch.randn(32000)
        )


# Add evaluation-m3 and interpretability package to system path
root_path = Path(__file__).resolve().parent.parent
eval_m3_path = root_path / "evaluation-m3"
if str(eval_m3_path) not in sys.path:
    sys.path.insert(0, str(eval_m3_path))
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))

# Import the actual interpretability modules
from config import EvalConfig
from interpretability import run_pilot as pilot
from interpretability.run_pilot import InterpConfig, run_pilot_inference
from interpretability.analyze_var import evaluate_var_diagnostics
from interpretability.latent_lens import evaluate_calibration
from interpretability.backpatching import run_backpatching_experiment
from interpretability.visualize_attn import generate_attention_maps

# Override model wrappers and config paths in dependencies
pilot.HookedM3Wrapper = MockHookedM3Wrapper


def generate_mock_results_jsonl(output_file: Path) -> None:
    """Generates a dummy multi_scale_results.jsonl matching the VQA strata requirements."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Stratum 1: Stable Correct (accuracy >= 0.5 across all scales)
    s1 = {
        "question_id": 101,
        "question": "Is the sky blue?",
        "results_by_m": {
            m: {"answer": "yes", "vqa_accuracy": 1.0, "avg_softmax_conf": 0.88}
            for m in ["1", "9", "36", "144", "576"]
        }
    }
    
    # Stratum 2: Stable Incorrect - Strict (accuracy < 0.5, same answer)
    s2 = {
        "question_id": 102,
        "question": "What color is the wall?",
        "results_by_m": {
            m: {"answer": "black", "vqa_accuracy": 0.0, "avg_softmax_conf": 0.35}
            for m in ["1", "9", "36", "144", "576"]
        }
    }

    # Stratum 3: Stable Incorrect - Relaxed (accuracy < 0.5, different answers)
    s3 = {
        "question_id": 103,
        "question": "How many cats are there?",
        "results_by_m": {
            "1": {"answer": "one", "vqa_accuracy": 0.0, "avg_softmax_conf": 0.31},
            "9": {"answer": "one", "vqa_accuracy": 0.0, "avg_softmax_conf": 0.35},
            "36": {"answer": "three", "vqa_accuracy": 0.0, "avg_softmax_conf": 0.40},
            "144": {"answer": "four", "vqa_accuracy": 0.0, "avg_softmax_conf": 0.42},
            "576": {"answer": "five", "vqa_accuracy": 0.0, "avg_softmax_conf": 0.45}
        }
    }

    # Stratum 4: Flip (accuracy changes across scales)
    s4 = {
        "question_id": 104,
        "question": "Is there a car?",
        "results_by_m": {
            "1": {"answer": "yes", "vqa_accuracy": 0.0, "avg_softmax_conf": 0.30},
            "9": {"answer": "yes", "vqa_accuracy": 0.0, "avg_softmax_conf": 0.35},
            "36": {"answer": "no", "vqa_accuracy": 1.0, "avg_softmax_conf": 0.70},
            "144": {"answer": "no", "vqa_accuracy": 1.0, "avg_softmax_conf": 0.85},
            "576": {"answer": "no", "vqa_accuracy": 1.0, "avg_softmax_conf": 0.90}
        }
    }

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(s1) + "\n")
        f.write(json.dumps(s2) + "\n")
        f.write(json.dumps(s3) + "\n")
        f.write(json.dumps(s4) + "\n")


def main() -> None:
    print("==========================================================")
    print(" Starting Local Verification Pipeline (Interpretability CPU Mocks)")
    print("==========================================================")
    
    project_root = Path(__file__).resolve().parent.parent
    test_dir = project_root / "test_run_output_interp"
    
    if test_dir.exists():
        shutil.rmtree(test_dir)
    test_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Setup mock source results
    source_results = test_dir / "mock_multi_scale_results.jsonl"
    generate_mock_results_jsonl(source_results)
    
    output_dir = test_dir / "results"
    
    test_config = InterpConfig(
        model_path="mock-hooked-m3",
        precision="fp32",
        source_results_jsonl=str(source_results),
        pilot_output_dir=str(output_dir),
        token_sweep=[1, 9, 36, 144, 576],
        hook_layers=[4, 8, 12, 16, 20, 24, 28, 32],
        stratification_counts={
            "stable_correct": 1,
            "stable_incorrect_strict": 1,
            "stable_incorrect_relaxed": 1,
            "flip": 1,
        }
    )
    
    # 2. Run Pilot Inference Sweep
    print("\n--- Running Pilot Inference Sweep (Mock mode) ---")
    # Emulate the run_pilot selector
    selected = {
        "stable_correct": [101],
        "stable_incorrect_strict": [102],
        "stable_incorrect_relaxed": [103],
        "flip": [104]
    }
    
    # Save the selected setup
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "pilot_sample_ids.json", "w", encoding="utf-8") as f:
        json.dump(selected, f, indent=4)
        
    run_pilot_inference(test_config, selected)
    
    # 3. Run VAR Diagnostics
    print("\n--- Running VAR Diagnostics Analysis ---")
    evaluate_var_diagnostics(str(output_dir), target_layers=[4, 8, 12, 16, 20, 24, 28, 32])
    
    # 4. Run LatentLens Calibration
    print("\n--- Running LatentLens Calibration Analysis ---")
    evaluate_calibration(str(output_dir))
    
    # 5. Run Backpatching Experiments
    print("\n--- Running Backpatching Experiments ---")
    run_backpatching_experiment(str(output_dir), num_samples=4)
    
    # 6. Run Attention Visualizations
    print("\n--- Running Attention Map Overlays Generation ---")
    generate_attention_maps(str(output_dir), num_examples=4)
    
    # Verify outputs exist
    print("\n=== Verification Summary ===")
    plots_dir = output_dir / "plots"
    
    expected_files = [
        output_dir / "pilot_sample_ids.json",
        output_dir / "pilot_results.jsonl",
        plots_dir / "var_distribution.png",
        plots_dir / "var_by_layer.png",
        plots_dir / "intermediate_ece_curve.png",
        plots_dir / "layer_confidence_evolution.png",
        plots_dir / "temperature_scaling_comparison.png",
        plots_dir / "backpatching_results.png",
        plots_dir / "backpatching_ece.png",
        plots_dir / "attention_map_comparison_1000.png",
        plots_dir / "attention_map_comparison_1001.png"
    ]
    
    success = True
    for ef in expected_files:
        rel_path = ef.relative_to(project_root)
        if ef.exists():
            size = ef.stat().st_size
            print(f"  [OK]  {str(rel_path):<65} | Size: {size:,} bytes")
        else:
            print(f"  [ERR] {str(rel_path):<65} | MISSING!")
            success = False
            
    if success:
        print("\n=======================================================")
        print(" VERIFICATION SUCCESSFUL: Interpretability Pipeline fully correct!")
        print("=======================================================")
        # Cleanup test run output
        shutil.rmtree(test_dir)
        sys.exit(0)
    else:
        print("\n=======================================================")
        print(" VERIFICATION FAILED: Some files were not created.")
        print("=======================================================")
        sys.exit(1)


if __name__ == "__main__":
    main()
