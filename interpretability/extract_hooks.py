"""Core PyTorch forward hooks manager for M3-LLaVA.

Extends the base M3LLaVAWrapper to capture intermediate hidden states,
attention weights, and construct token-to-position mapping.
"""

from __future__ import annotations

import sys
from pathlib import Path
import torch
from dataclasses import dataclass, field
from typing import Any, Generator
from contextlib import contextmanager

# RTX A4000 (Ampere SM86) — enable TF32 for matmul and cuDNN convolutions.
# TF32 gives ~8× throughput over FP32 with negligible accuracy loss for attention.
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Add evaluation-m3 to system path to load sibling modules
root_path = Path(__file__).resolve().parent.parent
eval_m3_path = root_path / "evaluation-m3"
if str(eval_m3_path) not in sys.path:
    sys.path.insert(0, str(eval_m3_path))

from model_wrapper import M3LLaVAWrapper
from config import EvalConfig
from llava.mm_utils import tokenizer_image_token

# Constants for LLaVA tokens
IMAGE_TOKEN_INDEX = -200

@dataclass
class TokenMap:
    """Maps token indices in the input sequence to semantic categories."""
    image_start: int
    image_end: int          # exclusive
    question_start: int
    question_end: int       # exclusive
    answer_start: int
    answer_end: int         # exclusive


@dataclass
class HookOutput:
    """Container for captured intermediate model states."""
    hidden_states: dict[int, torch.Tensor]       # layer_idx -> (seq_len, hidden_dim)
    attention_weights: dict[int, torch.Tensor]   # layer_idx -> (num_heads, seq_len, seq_len)
    token_map: TokenMap
    generated_text: str
    logits: torch.Tensor


@dataclass
class InterpConfig(EvalConfig):
    """Configuration class extended with interpretability-specific fields."""
    hook_layers: list[int] = field(default_factory=lambda: [4, 8, 12, 16, 20, 24, 28, 32])
    capture_attention: bool = True
    capture_hidden: bool = True
    pilot_samples: int = 1000
    pilot_output_dir: str = "results/pilot-interpretability"
    source_results_jsonl: str = "results/vlm-calibration-m3/results/multi_scale_results.jsonl"
    resume: bool = True
    debug: bool = False
    stratification_counts: dict[str, int] = field(default_factory=lambda: {
        "stable_correct": 400,
        "stable_incorrect_strict": 200,
        "stable_incorrect_relaxed": 200,
        "flip": 200,
    })

    def __post_init__(self) -> None:
        """Resolve pilot directories relative to project root without mutating parent paths."""
        import os
        cpu_count = os.cpu_count()
        if cpu_count is not None:
            self.num_workers = min(self.num_workers, max(1, cpu_count // 2))
            
        project_root = Path(__file__).resolve().parent.parent
        if not Path(self.pilot_output_dir).is_absolute():
            self.pilot_output_dir = str(project_root / self.pilot_output_dir)
        if not Path(self.source_results_jsonl).is_absolute():
            self.source_results_jsonl = str(project_root / self.source_results_jsonl)


class HookedM3Wrapper(M3LLaVAWrapper):
    """Wrapper for M3-LLaVA supporting dynamic PyTorch forward hooks for state extraction."""

    def __init__(self, model_path: str, model_base: str | None = None, precision: str = "bf16") -> None:
        # bf16 is Ampere-native (RTX A4000): same VRAM footprint as fp16 but with
        # wider dynamic range — avoids the gradient underflow issues that can affect
        # softmax entropy measurements in LogitLens.
        super().__init__(model_path=model_path, model_base=model_base, precision=precision)
        self._hooks: list[Any] = []
        self._captured_hidden: dict[int, torch.Tensor] = {}
        self._captured_attention: dict[int, torch.Tensor] = {}

    def register_hooks(
        self,
        layer_indices: list[int],
        capture_hidden: bool = True,
        capture_attention: bool = True
    ) -> None:
        """Register forward hooks on selected Transformer decoder layers.

        Args:
            layer_indices: 1-indexed list of layers to hook (e.g. 1 to 32).
            capture_hidden: Whether to hook hidden state outputs.
            capture_attention: Whether to hook self-attention outputs.
        """
        self.remove_hooks()
        
        # Access the underlying LLaMA model layers list
        # Typically model.model.layers for LlavaLlamaForCausalLM
        if not hasattr(self.model, "model") or not hasattr(self.model.model, "layers"):
            print("Warning: Model does not have layers attribute. Hook registration skipped (or mock mode).")
            return

        layers = self.model.model.layers
        num_layers = len(layers)

        for layer_idx in layer_indices:
            # Check 1-based indexing validity
            if layer_idx < 1 or layer_idx > num_layers:
                continue
            
            idx = layer_idx - 1  # 0-indexed internally

            # 1. Hidden State Hook
            if capture_hidden:
                def make_hidden_hook(l_idx: int):
                    def hidden_hook(module, inputs, outputs):
                        # HF layer outputs: (hidden_states, self_attns, present_key_values)
                        # non_blocking=True overlaps the D→H transfer with subsequent
                        # GPU compute (next layer forward), saving ~0.3 ms per hook on A4000.
                        if isinstance(outputs, tuple):
                            self._captured_hidden[l_idx] = outputs[0].detach().to(
                                "cpu", non_blocking=True
                            )
                        else:
                            self._captured_hidden[l_idx] = outputs.detach().to(
                                "cpu", non_blocking=True
                            )
                    return hidden_hook

                h_hook = layers[idx].register_forward_hook(make_hidden_hook(layer_idx))
                self._hooks.append(h_hook)

            # 2. Attention Weights Hook
            if capture_attention:
                def make_attention_hook(l_idx: int):
                    def attention_hook(module, inputs, outputs):
                        # LlamaAttention outputs: (attn_output, attn_weights, past_key_value)
                        # Note: attn_weights is only present if output_attentions=True is passed
                        if isinstance(outputs, tuple) and len(outputs) > 1 and outputs[1] is not None:
                            self._captured_attention[l_idx] = outputs[1].detach().to(
                                "cpu", non_blocking=True
                            )
                    return attention_hook

                if hasattr(layers[idx], "self_attn"):
                    a_hook = layers[idx].self_attn.register_forward_hook(make_attention_hook(layer_idx))
                    self._hooks.append(a_hook)

    def remove_hooks(self) -> None:
        """Remove all registered hooks, clear buffers, and release CUDA cache."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
        self.clear_captured()
        # Proactively release the CUDA memory allocator's cache so fragmented
        # blocks don't accumulate across the 1,000-sample pilot sweep.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def clear_captured(self) -> None:
        """Clear captured hidden states and attention weights."""
        self._captured_hidden.clear()
        self._captured_attention.clear()

    @contextmanager
    def hooked(
        self,
        layer_indices: list[int],
        capture_hidden: bool = True,
        capture_attention: bool = True
    ) -> Generator[HookedM3Wrapper, None, None]:
        """Context manager for registering and automatically cleaning up forward hooks."""
        self.register_hooks(layer_indices, capture_hidden, capture_attention)
        try:
            yield self
        finally:
            self.remove_hooks()

    @torch.inference_mode()
    def forward_with_hooks(
        self,
        image: Any,
        question: str,
        num_visual_tokens: int = 576,
        hook_layers: list[int] | None = None
    ) -> HookOutput:
        """Execute model forward pass with hooks active, returning captured hidden/attn states.

        Args:
            image: PIL Image input.
            question: Question string.
            num_visual_tokens: Matryoshka scale (m ∈ [1, 9, 36, 144, 576]).
            hook_layers: Layers to hook during this pass. Defaults to [4, 8, 12, 16, 20, 24, 28, 32].

        Returns:
            HookOutput containing captured states, generated answer, and positions.
        """
        if hook_layers is None:
            hook_layers = [4, 8, 12, 16, 20, 24, 28, 32]

        self.clear_captured()
        
        # Format and tokenize
        prompt = self.format_prompt(question)
        input_ids = (
            tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            .unsqueeze(0)
            .to(self.device)
        )
        
        # Find token map positions in input sequence
        # We need to map: image, question, answer.
        # Format is usually:
        # Prompt: [System info] <image> [Question text] Answer: [Generated Answer]
        input_ids_list = input_ids[0].tolist()
        
        # Locate image token index
        try:
            image_start = input_ids_list.index(IMAGE_TOKEN_INDEX)
            # In LLaVA, the <image> placeholder token expands to the actual number of visual tokens (e.g. 576)
            # during the visual projection step inside the model forward.
            # In input_ids, it's just a single token (-200).
            # The model replaces it with `num_visual_tokens` visual embeddings.
            # We must map token indices accounting for this expansion.
            image_end = image_start + num_visual_tokens
        except ValueError:
            image_start = 0
            image_end = 0

        # Preprocess image
        image_tensor, image_sizes = self.preprocess_image(image)
        
        # Run inference using the cached generator block, passing output_attentions=True
        # We register hooks before the forward execution if not already active
        was_registered = len(self._hooks) > 0
        if not was_registered:
            self.register_hooks(hook_layers, capture_hidden=True, capture_attention=True)
        
        # Use device.type ("cuda") as the autocast device string — passing the
        # full cuda:0 string causes a warning on some torch versions.
        autocast_device = self.device.type if hasattr(self.device, "type") else str(self.device).split(":")[0]
        with torch.amp.autocast(autocast_device, dtype=self.dtype):
            outputs = self.model.generate(
                input_ids,
                images=image_tensor,
                image_sizes=image_sizes,
                matryoshka_vis_token_scale=num_visual_tokens,
                do_sample=False,
                temperature=0.0,
                max_new_tokens=64,
                use_cache=True,
                output_attentions=True,
                output_hidden_states=True,
                return_dict_in_generate=True,
            )
            
        # Extract response answer text
        gen_sequence = outputs.sequences[0]
        num_generated = len(outputs.scores)
        gen_tokens = gen_sequence[-num_generated:].tolist()
        answer_text = self.tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
        
        # Calculate precise token map coordinates (adjusted for visual token expansion)
        # Sequence length in model input space is: len(input_ids) - 1 + num_visual_tokens
        # Prefix before image: input_ids_list[:image_start]
        # Image region: image_start to image_start + num_visual_tokens
        # Suffix after image (question): image_start + num_visual_tokens onwards
        total_prefill_len = len(input_ids_list) - 1 + num_visual_tokens
        
        question_start = image_end
        question_end = total_prefill_len
        answer_start = total_prefill_len
        answer_end = total_prefill_len + num_generated
        
        t_map = TokenMap(
            image_start=image_start,
            image_end=image_end,
            question_start=question_start,
            question_end=question_end,
            answer_start=answer_start,
            answer_end=answer_end
        )
        
        # Copy captured dicts to return HookOutput, detach hook outputs
        captured_hidden = {k: v for k, v in self._captured_hidden.items()}
        captured_attn = {k: v for k, v in self._captured_attention.items()}
        
        # Warning if attention is not captured but expected
        if not captured_attn:
            import logging
            logging.getLogger("VLM_Interpretability").warning(
                "Attention weights were not captured. Attention hooks may capture nothing if output_attentions=True is not active or supported by the decoder layers."
            )
        
        if not was_registered:
            self.remove_hooks()
        
        # Extract raw logit tensor at the last step
        logits = outputs.scores[-1][0].detach().cpu()
        
        return HookOutput(
            hidden_states=captured_hidden,
            attention_weights=captured_attn,
            token_map=t_map,
            generated_text=answer_text,
            logits=logits
        )
