"""
Abstraction layer for loading and running inference with the Matryoshka Multimodal Model (M3-LLaVA).

Handles device-agnostic configuration, prompt formatting, image processing,
and extracting token-level confidence scores (both logprobs and softmax).
"""

from __future__ import annotations

import sys
from pathlib import Path
import re
import torch
from typing import Any
from PIL import Image

# Ensure the cloned matryoshka-mm repository is in the path
repo_path = Path(__file__).resolve().parent.parent / "matryoshka-mm"
if str(repo_path) not in sys.path:
    sys.path.insert(0, str(repo_path))

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path


class M3LLaVAWrapper:
    """Wrapper class for M3-LLaVA models enabling elastic token inference via average-pooling grids."""

    def __init__(self, model_path: str, model_base: str | None = None, precision: str = "fp16") -> None:
        """Load the M3-LLaVA model and tokenizer.

        Args:
            model_path: Path to the model checkpoint or HuggingFace repo ID.
            model_base: Path to base model if loading LoRA weights.
            precision: FP precision to use ("fp16", "bf16", "fp32").
        """
        disable_torch_init()
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Initializing model on device: {self.device}")
        
        # Resolve dtype
        if precision == "fp16":
            self.dtype = torch.float16
        elif precision == "bf16":
            self.dtype = torch.bfloat16
        else:
            self.dtype = torch.float32
            
        # Extract model name
        self.model_name = get_model_name_from_path(model_path)
        
        # Load tokenizer, model, and image processor
        tokenizer, model, image_processor, context_len = load_pretrained_model(
            model_path=model_path,
            model_base=model_base,
            model_name=self.model_name,
            device_map="auto" if torch.cuda.is_available() else "cpu",
            torch_dtype=self.dtype
        )
        
        self.tokenizer = tokenizer
        self.model = model
        self.image_processor = image_processor
        self.context_len = context_len
        
        # Set conversation mode based on model name
        if "llama-2" in self.model_name.lower():
            self.conv_mode = "llava_llama_2"
        elif "mistral" in self.model_name.lower():
            self.conv_mode = "mistral_instruct"
        elif "v1.6" in self.model_name.lower():
            self.conv_mode = "chatml_direct"
        elif "v1" in self.model_name.lower():
            self.conv_mode = "llava_v1"
        elif "mpt" in self.model_name.lower():
            self.conv_mode = "mpt"
        else:
            self.conv_mode = "llava_v0"
            
        print(f"Model loaded successfully. Conversation template: {self.conv_mode}")

    def preprocess_image(self, image: Image.Image) -> tuple[torch.Tensor, list[tuple[int, int]]]:
        """Preprocess PIL image into torch tensor and size metadata.

        Args:
            image: PIL image object.

        Returns:
            Tuple of (preprocessed_image_tensor, image_sizes_list).
        """
        image_sizes = [image.size]
        image_tensor = process_images(
            [image],
            self.image_processor,
            self.model.config
        ).to(self.device, dtype=self.dtype)
        return image_tensor, image_sizes

    def format_prompt(self, question: str) -> str:
        """Format the input question using the correct conversation template.

        Args:
            question: Raw input question.

        Returns:
            Formatted prompt string.
        """
        image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
        
        # Align image tokens in the question
        if IMAGE_PLACEHOLDER in question:
            if self.model.config.mm_use_im_start_end:
                qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, question)
            else:
                qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, question)
        else:
            if self.model.config.mm_use_im_start_end:
                qs = image_token_se + "\n" + question
            else:
                qs = DEFAULT_IMAGE_TOKEN + "\n" + question
        
        # Append short answer instruction to match VQAv2 ground-truth annotations
        qs = qs + "\nAnswer the question using a single word or phrase."
                
        # Build prompt from conversation template
        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        return conv.get_prompt()

    @torch.inference_mode()
    def generate_with_confidence(
        self,
        image: Image.Image,
        question: str,
        num_visual_tokens: int = 576,
        max_new_tokens: int = 64,
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        """Run inference at a specific token depth and extract confidence metrics.

        Args:
            image: Input PIL image.
            question: Input question.
            num_visual_tokens: Number of query tokens to use (m ∈ [1, 9, 36, 144, 576]).
            max_new_tokens: Maximum number of tokens to generate.
            temperature: Generation temperature (0.0 for greedy decoding).

        Returns:
            Dict containing generation results and confidence details.
        """
        # 1. Format prompt and tokenize
        prompt = self.format_prompt(question)
        input_ids = (
            tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            .unsqueeze(0)
            .to(self.device)
        )
        
        # 2. Preprocess image
        image_tensor, image_sizes = self.preprocess_image(image)
        
        # 3. Model generation and evaluation
        return self._generate_from_cached(
            image_tensor=image_tensor,
            image_sizes=image_sizes,
            input_ids=input_ids,
            num_visual_tokens=num_visual_tokens,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )

    @torch.inference_mode()
    def _generate_from_cached(
        self,
        image_tensor: torch.Tensor,
        image_sizes: list[tuple[int, int]],
        input_ids: torch.Tensor,
        num_visual_tokens: int = 576,
        max_new_tokens: int = 64,
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        """Run inference using pre-computed image tensor and input_ids."""
        with torch.amp.autocast(str(self.device), dtype=self.dtype):
            # Pass matryoshka_vis_token_scale to generate() to control the pooling factor
            outputs = self.model.generate(
                input_ids,
                images=image_tensor,
                image_sizes=image_sizes,
                matryoshka_vis_token_scale=num_visual_tokens,
                do_sample=True if temperature > 0.0 else False,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                output_scores=True,
                return_dict_in_generate=True,
            )
        
        # 4. Decode text
        gen_sequence = outputs.sequences[0]
        num_generated = len(outputs.scores)  # number of decoding steps
        gen_tokens = gen_sequence[-num_generated:].tolist()  # (num_generated,)
        
        # 5. Extract token-level logprobs and softmax probs
        token_details = []
        total_log_prob = 0.0
        total_prob_product = 1.0
        
        # EOS token ID
        eos_token_id = self.tokenizer.eos_token_id
        
        for i, logits_step in enumerate(outputs.scores):
            logits = logits_step[0]  # -> (vocab_size,)
            
            probs = torch.softmax(logits, dim=-1)
            log_probs = torch.log_softmax(logits, dim=-1)
            
            token_id = gen_tokens[i]
            token_text = self.tokenizer.decode([token_id])
            
            token_prob = probs[token_id].item()
            token_logprob = log_probs[token_id].item()
            
            token_details.append({
                "step": i,
                "token_id": token_id,
                "token_text": token_text,
                "prob": token_prob,
                "logprob": token_logprob,
                "is_eos": token_id == eos_token_id
            })
            
            # Exclude EOS from main joint probability/logprob sums if it's the last token
            if token_id != eos_token_id or i != num_generated - 1:
                total_log_prob += token_logprob
                total_prob_product *= token_prob
                
        # Clean answer text
        answer_text = self.tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
        
        # Calculate averages
        effective_len = len([t for t in token_details if not t["is_eos"]])
        if effective_len == 0:
            effective_len = len(token_details) or 1
            
        avg_log_prob = total_log_prob / effective_len
        avg_softmax_conf = total_prob_product ** (1.0 / effective_len)
        
        return {
            "answer": answer_text,
            "log_prob": total_log_prob,
            "avg_log_prob": avg_log_prob,
            "softmax_conf": total_prob_product,
            "avg_softmax_conf": avg_softmax_conf,
            "generated_tokens": gen_tokens,
            "token_details": token_details,
        }

    def sweep(
        self,
        image: Image.Image,
        question: str,
        token_counts: list[int],
        max_new_tokens: int = 64,
        temperature: float = 0.0,
    ) -> dict[int, dict[str, Any]]:
        """Run inference across multiple token counts for the same image and question."""
        results = {}
        for m in token_counts:
            res = self.generate_with_confidence(
                image=image,
                question=question,
                num_visual_tokens=m,
                max_new_tokens=max_new_tokens,
                temperature=temperature
            )
            results[m] = res
        return results

    def sweep_optimized(
        self,
        image: Image.Image,
        question: str,
        token_counts: list[int],
        max_new_tokens: int = 64,
        temperature: float = 0.0,
    ) -> dict[int, dict[str, Any]]:
        """Optimized sweep: preprocess image and tokenize prompt ONCE."""
        # 1. Preprocess image once
        image_tensor, image_sizes = self.preprocess_image(image)
        
        # 2. Tokenize prompt once
        prompt = self.format_prompt(question)
        input_ids = (
            tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            .unsqueeze(0)
            .to(self.device)
        )
        
        # 3. Loop only over token counts
        results = {}
        for m in token_counts:
            res = self._generate_from_cached(
                image_tensor=image_tensor,
                image_sizes=image_sizes,
                input_ids=input_ids,
                num_visual_tokens=m,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
            results[m] = res
        return results


# Set alias MQTLLaVAWrapper to maintain backward compatibility
MQTLLaVAWrapper = M3LLaVAWrapper
