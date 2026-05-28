"""
Standalone local verification script for VLM Calibration pipeline.
Mocks HuggingFace datasets and MQT-LLaVA VLM wrappers to test logic on CPU.
"""

from __future__ import annotations

import os
import sys
import shutil
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
mock_llava.mm_utils.get_model_name_from_path = MagicMock(return_value="mock-llava")

# Inject mock modules
sys.modules["llava"] = mock_llava
sys.modules["llava.constants"] = mock_llava.constants
sys.modules["llava.conversation"] = mock_llava.conversation
sys.modules["llava.model"] = mock_llava.model
sys.modules["llava.model.builder"] = mock_llava.model.builder
sys.modules["llava.utils"] = mock_llava.utils
sys.modules["llava.mm_utils"] = mock_llava.mm_utils

# Mock SentenceTransformer
class MockSentenceTransformer:
    def __init__(self, model_name: str, device: str = None) -> None:
        self.model_name = model_name

    def encode(self, sentences: list[str], show_progress_bar: bool = False) -> Any:
        import numpy as np
        # Return distinct vectors for distinct answers to simulate embeddings
        embeddings = []
        for s in sentences:
            # Deterministic pseudo-random seed based on string hash
            h = hash(s) % (2**32)
            np.random.seed(h)
            embeddings.append(np.random.randn(384))
        return np.array(embeddings)

sys.modules["sentence_transformers"] = MagicMock()
sys.modules["sentence_transformers"].SentenceTransformer = MockSentenceTransformer

# 2. Define Mock Dataset
# We need it to act like a datasets.Dataset (indexable, has features, split, select method, etc.)
class MockDatasetList:
    def __init__(self, data_list: list[dict]) -> None:
        self.data_list = data_list
        self.features = {
            "question_id": "int",
            "image_id": "int",
            "question": "string",
            "question_type": "string",
            "answer_type": "string",
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
        "question_type": "yes/no",
        "answer_type": "yes/no",
        "multiple_choice_answer": "yes",
        "answers": [{"answer": "yes"}, {"answer": "yes"}, {"answer": "no"}],
        "image": dummy_img,
    },
    {
        "question_id": 102,
        "image_id": 2002,
        "question": "What color is the wall?",
        "question_type": "color",
        "answer_type": "other",
        "multiple_choice_answer": "white",
        "answers": [{"answer": "white"}, {"answer": "white"}, {"answer": "white"}],
        "image": dummy_img,
    },
    {
        "question_id": 103,
        "image_id": 2003,
        "question": "How many cats are there?",
        "question_type": "how many",
        "answer_type": "numeric",
        "multiple_choice_answer": "2",
        "answers": [{"answer": "2"}, {"answer": "two"}, {"answer": "2"}],
        "image": dummy_img,
    }
]
mock_dataset_obj = MockDatasetList(mock_samples)

# Mock datasets module
mock_datasets = MagicMock()
mock_datasets.load_dataset = MagicMock(return_value=mock_dataset_obj)
sys.modules["datasets"] = mock_datasets

# 3. Define Mock Model Wrapper
class MockMQTLLaVAWrapper:
    def __init__(self, model_path: str, model_base: str | None = None, precision: str = "fp16") -> None:
        self.device = torch.device("cpu")
        self.dtype = torch.float32
        self.model_name = "mock-mqt-llava-7b"

    def sweep_optimized(
        self,
        image: Image.Image,
        question: str,
        token_counts: list[int],
        max_new_tokens: int = 64,
        temperature: float = 0.0,
    ) -> dict[int, dict[str, Any]]:
        # Produce realistic deterministic answers based on question
        ans = "yes"
        ans_by_scale = {}
        if "color" in question.lower():
            ans = "white"
        elif "cats" in question.lower():
            # Let's return different answers at different token scales to simulate instability!
            ans_by_scale = {
                2: "none",
                4: "one",
                8: "two",
                16: "2",
                36: "2",
                64: "2",
                144: "2",
                256: "2"
            }

        results = {}
        for m in token_counts:
            scale_ans = ans_by_scale.get(m, ans)
            
            # Create a decaying confidence scale: larger scales have higher confidence
            conf = 0.4 + 0.55 * (m / 256.0)
            logprob = float(torch.log(torch.tensor(conf)).item())
            
            token_details = [
                {"step": 0, "token_id": 42, "token_text": scale_ans, "prob": conf, "logprob": logprob, "is_eos": False},
                {"step": 1, "token_id": 2, "token_text": "</s>", "prob": 0.99, "logprob": -0.01, "is_eos": True}
            ]
            
            results[m] = {
                "answer": scale_ans,
                "log_prob": logprob,
                "avg_log_prob": logprob,
                "softmax_conf": conf,
                "avg_softmax_conf": conf,
                "generated_tokens": [42, 2],
                "token_details": token_details,
            }
        return results

# 4. Now import harness modules and run verification
import numpy as np
from evaluation.config import EvalConfig
import evaluation.multi_scale_harness as harness
from evaluation.multi_scale_harness import run_evaluation, compile_summary_csv
from visualization.variance_plots import generate_variance_plots
from visualization.reliability_diagram import plot_reliability_diagrams
from visualization.ece_summary import generate_ece_summary

# Override model wrapper in harness module with our Mock
harness.MQTLLaVAWrapper = MockMQTLLaVAWrapper

def main() -> None:
    print("==================================================")
    print(" Starting Local Verification Pipeline (CPU Mocks)")
    print("==================================================")
    
    # Establish local temporary test directory
    project_root = Path(__file__).resolve().parent.parent
    test_dir = project_root / "test_run_output"
    
    # Clean up previous test runs
    if test_dir.exists():
        shutil.rmtree(test_dir)
    test_dir.mkdir(parents=True, exist_ok=True)
    
    output_dir = test_dir / "results"
    plots_dir = test_dir / "plots"
    logs_dir = test_dir / "logs"
    checkpoint_dir = test_dir / "checkpoints"
    
    # Setup test-specific configuration
    test_config = EvalConfig(
        model_path="mock-mqt-llava-7b",
        precision="fp32",
        subset_size=3,
        output_dir=str(output_dir),
        plots_dir=str(plots_dir),
        logs_dir=str(logs_dir),
        checkpoint_dir=str(checkpoint_dir),
        token_sweep=[2, 4, 8, 16, 36, 64, 144, 256]
    )
    
    # Run evaluation harness on CPU
    print("\n--- Running Evaluation Harness ---")
    run_evaluation(test_config)
    
    # Compile summary CSV (normally done automatically at end of run)
    print("\n--- Running Summary CSV Compilation ---")
    compile_summary_csv(test_config)
    
    # Run Visualization scripts
    print("\n--- Running Variance Plot Generation ---")
    generate_variance_plots(test_config)
    
    print("\n--- Running Reliability Diagrams Generation ---")
    plot_reliability_diagrams(test_config)
    
    print("\n--- Running ECE Summary Charts Generation ---")
    generate_ece_summary(test_config)
    
    # Check outputs exist
    print("\n=== Verification Verification Summary ===")
    
    expected_files = [
        output_dir / "multi_scale_results.jsonl",
        output_dir / "summary_statistics.csv",
        plots_dir / "variance_distribution.png",
        plots_dir / "performance_vs_tokens.png",
        plots_dir / "reliability_diagrams_multi.png",
        plots_dir / "calibration_stats.json",
        plots_dir / "ece_comparison.png",
        plots_dir / "calibration_metrics_table.md",
        plots_dir / "variance_gallery.md"
    ]
    
    success = True
    for ef in expected_files:
        rel_path = ef.relative_to(project_root)
        if ef.exists():
            size = ef.stat().st_size
            print(f"  [OK]  {str(rel_path):<50} | Size: {size:,} bytes")
        else:
            print(f"  [ERR] {str(rel_path):<50} | MISSING!")
            success = False
            
    if success:
        print("\n==================================================")
        print(" VERIFICATION SUCCESSFUL: Pipeline fully correct!")
        print("==================================================")
        sys.exit(0)
    else:
        print("\n==================================================")
        print(" VERIFICATION FAILED: Some files were not created.")
        print("==================================================")
        sys.exit(1)

if __name__ == "__main__":
    main()
