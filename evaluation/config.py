"""
Centralized configuration for the VLM Calibration evaluation pipeline.

All tunable parameters for the multi-scale evaluation harness, model loading,
dataset selection, and visualization are defined here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class EvalConfig:
    """Configuration for the multi-scale VLM calibration evaluation."""

    # ── Model ────────────────────────────────────────────────────────────
    model_path: str = "gordonhu/MQT-LLaVA-7b"
    model_name: str = "mqt-llava"
    precision: str = "fp16"  # "fp16" | "bf16" | "fp32"

    # ── Token Sweep ──────────────────────────────────────────────────────
    # MQT-LLaVA supports up to 256 learnable query tokens.
    # Tail-token dropping during training ensures the first m tokens
    # carry the most important visual information (Matryoshka structure).
    token_sweep: list[int] = field(
        default_factory=lambda: [2, 4, 8, 16, 36, 64, 144, 256]
    )

    # ── Dataset ──────────────────────────────────────────────────────────
    dataset_name: str = "lmms-lab/VQAv2"
    dataset_split: str = "all"
    subset_size: int | None = None  # None = full dataset (~1.1M samples)

    # ── Output ───────────────────────────────────────────────────────────
    output_dir: str = "/workspace/vlm-calibration/results"
    plots_dir: str = "/workspace/vlm-calibration/plots"
    logs_dir: str = "/workspace/vlm-calibration/logs"
    checkpoint_dir: str = "/workspace/vlm-calibration/checkpoints"

    # ── Harness Control ──────────────────────────────────────────────────
    checkpoint_interval: int = 500     # save results every N samples
    archive_interval: int = 2000      # archive snapshots of plots/stats every N samples
    batch_size: int = 1                # inference batch size (1 for consistency)
    num_workers: int = 8               # dataloader workers (16 vCPU available)
    pin_memory: bool = True            # faster CPU→GPU transfer
    prefetch_factor: int = 2           # batches to prefetch per worker
    seed: int = 42                     # reproducibility seed

    # ── Embedding Similarity ─────────────────────────────────────────────
    # Used for computing answer stability across token depths.
    # all-MiniLM-L6-v2 is a fast, lightweight sentence encoder.
    embedding_model: str = "all-MiniLM-L6-v2"

    # ── Visualization ────────────────────────────────────────────────────
    gallery_top_k: int = 20            # top-K samples for variance galleries
    ece_num_bins: int = 15             # number of bins for ECE / reliability diagrams
    figure_dpi: int = 150              # resolution for saved figures
    figure_format: str = "png"         # "png" | "pdf" | "svg"

    # ── Generation ───────────────────────────────────────────────────────
    max_new_tokens: int = 64           # max tokens to generate per answer
    temperature: float = 0.0           # greedy decoding (deterministic)
    do_sample: bool = False            # no sampling for reproducibility

    def results_jsonl_path(self) -> Path:
        """Path to the main results JSONL file."""
        return Path(self.output_dir) / "multi_scale_results.jsonl"

    def summary_csv_path(self) -> Path:
        """Path to the per-sample summary statistics CSV."""
        return Path(self.output_dir) / "summary_statistics.csv"

    def checkpoint_path(self, sample_idx: int) -> Path:
        """Path to a specific checkpoint file."""
        return Path(self.checkpoint_dir) / f"checkpoint_{sample_idx:08d}.jsonl"

    def latest_checkpoint_path(self) -> Path:
        """Path to the latest checkpoint marker."""
        return Path(self.checkpoint_dir) / "latest.txt"

    def __post_init__(self) -> None:
        """Resolve directories dynamically based on environment."""
        workspace_path = Path("/workspace")
        if not workspace_path.exists():
            project_root = Path(__file__).resolve().parent.parent
            if self.output_dir == "/workspace/vlm-calibration/results":
                self.output_dir = str(project_root / "results")
            if self.plots_dir == "/workspace/vlm-calibration/plots":
                self.plots_dir = str(project_root / "plots")
            if self.logs_dir == "/workspace/vlm-calibration/logs":
                self.logs_dir = str(project_root / "logs")
            if self.checkpoint_dir == "/workspace/vlm-calibration/checkpoints":
                self.checkpoint_dir = str(project_root / "checkpoints")

        # Dynamically scale num_workers to half of available vCPUs to prevent CPU thrashing
        import os
        cpu_count = os.cpu_count()
        if cpu_count is not None:
            self.num_workers = min(self.num_workers, max(1, cpu_count // 2))

    def ensure_dirs(self) -> None:
        """Create all output directories if they don't exist."""
        for d in [self.output_dir, self.plots_dir, self.logs_dir, self.checkpoint_dir]:
            Path(d).mkdir(parents=True, exist_ok=True)



# Default configuration instance
DEFAULT_CONFIG = EvalConfig()
