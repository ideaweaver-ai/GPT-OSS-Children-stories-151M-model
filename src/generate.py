"""
Advanced Text Generation Script for GPT-OSS Children's Stories
Supports multiple sampling strategies and advanced features
"""

import torch
import tiktoken
import argparse
import json
import os
import time
from typing import Optional, List, Dict, Any
from transformers import GPT2TokenizerFast

from model.gpt_oss_advanced import GPTOSSAdvanced, GPTOSSAdvancedConfig
from config import ModelConfig, GenerationConfig


class AdvancedTextGenerator:
    """Advanced text generator with multiple sampling strategies"""
    
    def __init__(self, model_path: str, device: str = "auto"):
        self.device = self._setup_device(device)
        self.model, self.model_config = self._load_model(model_path)
        self.tokenizer = self._setup_tokenizer()
        
        print(f"Generator initialized:")
        print(f"  Device: {self.device}")
        print(f"  Model parameters: {self.model.get_num_params():,}")
        print(f"  Vocab size: {self.model_config.vocab_size}")
    
    def _setup_device(self, device: str) -> torch.device:
        """Setup compute device"""
        if device == "auto":
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")
        else:
            device = torch.device(device)
        
        print(f"Using device: {device}")
        return device
    
    def _load_model(self, model_path: str) -> tuple:
        """Load model from checkpoint"""
        print(f"Loading model from {model_path}")
        
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Extract model config
        if 'model_config' in checkpoint:
            config_dict = checkpoint['model_config']
        else:
            # Fallback to default config
            config_dict = {}
        
        # Create GPTOSSAdvancedConfig
        gpt_config = GPTOSSAdvancedConfig(**config_dict)
        
        # Create model
        model = GPTOSSAdvanced(gpt_config)
        
        # Load state dict
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model = model.to(self.device)
        model.eval()
        
        return model, gpt_config
    
    def _setup_tokenizer(self):
        """Setup tokenizer based on model configuration"""
        # Try to determine the correct tokenizer based on vocab size
        if hasattr(self.model_config, 'vocab_size') and self.model_config.vocab_size == 201088:
            # GPT-OSS model - use harmony tokenizer
            try:
                # Try to use the harmony tokenizer from openai-harmony package
                try:
                    from openai_harmony import load_harmony_encoding, HarmonyEncodingName
                    tokenizer = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
                    print("Using GPT-OSS harmony tokenizer")
                    return tokenizer
                except ImportError:
                    print("openai-harmony not available, falling back to o200k_base")
                    tokenizer = tiktoken.get_encoding("o200k_base")
                    print("Using tiktoken o200k_base tokenizer (GPT-OSS compatible)")
                    return tokenizer
            except Exception as e:
                print(f"Failed to load GPT-OSS tokenizer: {e}")
                print("WARNING: Falling back to GPT-2 tokenizer - this may cause issues")
        
        # Fallback to GPT-2 tokenizer
        try:
            tokenizer = tiktoken.get_encoding("gpt2")
            print("Using tiktoken GPT-2 tokenizer")
            return tokenizer
        except Exception:
            # Final fallback to HuggingFace
            tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            print("Using HuggingFace GPT-2 tokenizer")
            return tokenizer
    
    def encode_text(self, text: str) -> torch.Tensor:
        """Encode text to token IDs"""
        if hasattr(self.tokenizer, 'encode'):
            # tiktoken
            tokens = self.tokenizer.encode(text)
        else:
            # HuggingFace
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
        
        return torch.tensor(tokens, dtype=torch.long, device=self.device).unsqueeze(0)
    
    def decode_tokens(self, tokens: torch.Tensor) -> str:
        """Decode token IDs to text"""
        if tokens.dim() > 1:
            tokens = tokens.squeeze()
        
        token_list = tokens.cpu().tolist()
        
        if hasattr(self.tokenizer, 'decode'):
            # tiktoken or HuggingFace
            try:
                text = self.tokenizer.decode(token_list)
            except Exception:
                # Handle potential decoding errors
                text = self.tokenizer.decode(token_list, errors='ignore')
        else:
            text = "".join([str(t) for t in token_list])
        
        return text
    
    def apply_repetition_penalty(self, logits: torch.Tensor, input_ids: torch.Tensor, penalty: float = 1.1) -> torch.Tensor:
        """Apply repetition penalty to logits"""
        if penalty == 1.0:
            return logits
        
        # Get unique tokens in the input
        unique_tokens = torch.unique(input_ids)
        
        # Apply penalty
        for token in unique_tokens:
            if logits[0, token] > 0:
                logits[0, token] /= penalty
            else:
                logits[0, token] *= penalty
        
        return logits
    
    def top_k_top_p_filtering(self, logits: torch.Tensor, top_k: int = 0, top_p: float = 1.0) -> torch.Tensor:
        """Apply top-k and top-p (nucleus) filtering"""
        if top_k > 0:
            # Top-k filtering
            top_k = min(top_k, logits.size(-1))
            values, _ = torch.topk(logits, top_k)
            min_value = values[:, -1].unsqueeze(-1)
            logits = torch.where(logits < min_value, torch.full_like(logits, -float('inf')), logits)
        
        if top_p < 1.0:
            # Top-p (nucleus) filtering
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            # Scatter sorted tensors to original indexing
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits = logits.masked_fill(indices_to_remove, -float('inf'))
        
        return logits
    
    @torch.no_grad()
    def generate(
        self,
        prompt: str = "",
        max_new_tokens: int = 200,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
        do_sample: bool = True,
        num_return_sequences: int = 1,
        seed: Optional[int] = None
    ) -> List[str]:
        """Generate text with advanced sampling"""
        
        if seed is not None:
            torch.manual_seed(seed)
        
        # Encode prompt
        if prompt:
            input_ids = self.encode_text(prompt)
        else:
            # Start with BOS token
            input_ids = torch.tensor([[self.tokenizer.encode("<|endoftext|>")[0]]], 
                                   dtype=torch.long, device=self.device)
        
        original_length = input_ids.shape[1]
        
        # Generate multiple sequences if requested
        all_sequences = []
        
        for _ in range(num_return_sequences):
            current_input_ids = input_ids.clone()
            
            for _ in range(max_new_tokens):
                # Crop input if it exceeds model's context length
                if current_input_ids.shape[1] > self.model_config.block_size:
                    current_input_ids = current_input_ids[:, -self.model_config.block_size:]
                
                # Forward pass
                logits, _ = self.model(current_input_ids)
                logits = logits[:, -1, :]  # Get last token logits
                
                # Apply repetition penalty
                if repetition_penalty != 1.0:
                    logits = self.apply_repetition_penalty(logits, current_input_ids, repetition_penalty)
                
                # Apply temperature
                if temperature != 1.0:
                    logits = logits / temperature
                
                # Apply top-k and top-p filtering
                if do_sample:
                    filtered_logits = self.top_k_top_p_filtering(logits, top_k, top_p)
                    probs = F.softmax(filtered_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    # Greedy decoding
                    next_token = torch.argmax(logits, dim=-1, keepdim=True)
                
                # Append token
                current_input_ids = torch.cat([current_input_ids, next_token], dim=-1)
                
                # Check for EOS token
                if hasattr(self.tokenizer, 'eos_token_id') and next_token.item() == self.tokenizer.eos_token_id:
                    break
                elif next_token.item() == 50256:  # GPT-2 EOS token
                    break
            
            # Decode generated sequence
            generated_tokens = current_input_ids[0, original_length:]
            generated_text = self.decode_tokens(generated_tokens)
            all_sequences.append(generated_text)
        
        return all_sequences
    
    def interactive_generation(self):
        """Interactive text generation"""
        print("\n" + "="*60)
        print("GPT-OSS Advanced Children's Stories - Interactive Mode")
        print("="*60)
        print("Enter prompts to generate stories. Type 'quit' to exit.")
        print("Type 'config' to change generation settings.")
        print("-"*60)
        
        # Default generation config
        gen_config = {
            'max_new_tokens': 200,
            'temperature': 0.8,
            'top_k': 50,
            'top_p': 0.9,
            'repetition_penalty': 1.1,
            'num_return_sequences': 1
        }
        
        while True:
            try:
                prompt = input("\nPrompt: ").strip()
                
                if prompt.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break
                
                if prompt.lower() == 'config':
                    print("\nCurrent settings:")
                    for key, value in gen_config.items():
                        print(f"  {key}: {value}")
                    
                    print("\nEnter new values (press Enter to keep current):")
                    for key in gen_config.keys():
                        new_value = input(f"  {key} [{gen_config[key]}]: ").strip()
                        if new_value:
                            try:
                                if key in ['max_new_tokens', 'top_k', 'num_return_sequences']:
                                    gen_config[key] = int(new_value)
                                else:
                                    gen_config[key] = float(new_value)
                            except ValueError:
                                print(f"Invalid value for {key}, keeping current value")
                    continue
                
                if not prompt:
                    prompt = "Once upon a time"
                
                print(f"\nGenerating story with prompt: '{prompt}'...")
                start_time = time.time()
                
                stories = self.generate(prompt=prompt, **gen_config)
                
                generation_time = time.time() - start_time
                
                print(f"\n{'='*60}")
                print(f"Generated in {generation_time:.2f} seconds")
                print(f"{'='*60}")
                
                for i, story in enumerate(stories):
                    if len(stories) > 1:
                        print(f"\n--- Story {i+1} ---")
                    print(f"\n{prompt}{story}")
                    print(f"\n{'-'*60}")
                
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error during generation: {e}")
    
    def benchmark_generation(self, num_samples: int = 10, max_tokens: int = 100):
        """Benchmark generation speed"""
        print(f"\nBenchmarking generation speed...")
        print(f"Samples: {num_samples}, Max tokens per sample: {max_tokens}")
        
        prompts = [
            "Once upon a time",
            "In a magical forest",
            "There was a brave little mouse",
            "Long ago in a distant kingdom",
            "A young girl discovered"
        ]
        
        total_tokens = 0
        total_time = 0
        
        for i in range(num_samples):
            prompt = prompts[i % len(prompts)]
            
            start_time = time.time()
            stories = self.generate(
                prompt=prompt,
                max_new_tokens=max_tokens,
                temperature=0.8,
                do_sample=True
            )
            end_time = time.time()
            
            generation_time = end_time - start_time
            story_tokens = len(self.encode_text(stories[0]))
            
            total_tokens += story_tokens
            total_time += generation_time
            
            print(f"Sample {i+1}: {story_tokens} tokens in {generation_time:.2f}s "
                  f"({story_tokens/generation_time:.1f} tokens/s)")
        
        avg_tokens_per_sec = total_tokens / total_time
        print(f"\nAverage: {avg_tokens_per_sec:.1f} tokens/second")
        print(f"Total tokens: {total_tokens}")
        print(f"Total time: {total_time:.2f} seconds")


def main():
    parser = argparse.ArgumentParser(description="Generate children's stories using GPT-OSS Advanced")
    
    parser.add_argument("--model-path", type=str, required=True,
                       help="Path to model checkpoint")
    parser.add_argument("--prompt", type=str, default="Once upon a time",
                       help="Text prompt for generation")
    parser.add_argument("--max-tokens", type=int, default=200,
                       help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8,
                       help="Sampling temperature")
    parser.add_argument("--top-k", type=int, default=50,
                       help="Top-k filtering")
    parser.add_argument("--top-p", type=float, default=0.9,
                       help="Top-p (nucleus) filtering")
    parser.add_argument("--repetition-penalty", type=float, default=1.1,
                       help="Repetition penalty")
    parser.add_argument("--num-samples", type=int, default=1,
                       help="Number of samples to generate")
    parser.add_argument("--device", type=str, default="auto",
                       choices=["auto", "cuda", "cpu", "mps"],
                       help="Device to use for generation")
    parser.add_argument("--interactive", action="store_true",
                       help="Run in interactive mode")
    parser.add_argument("--benchmark", action="store_true",
                       help="Run generation benchmark")
    parser.add_argument("--seed", type=int, default=None,
                       help="Random seed for reproducible generation")
    
    args = parser.parse_args()
    
    # Create generator
    generator = AdvancedTextGenerator(args.model_path, args.device)
    
    if args.interactive:
        generator.interactive_generation()
    elif args.benchmark:
        generator.benchmark_generation()
    else:
        # Single generation
        print(f"Generating story with prompt: '{args.prompt}'")
        print(f"Settings: temp={args.temperature}, top_k={args.top_k}, top_p={args.top_p}")
        print("-" * 60)
        
        start_time = time.time()
        stories = generator.generate(
            prompt=args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            num_return_sequences=args.num_samples,
            seed=args.seed
        )
        generation_time = time.time() - start_time
        
        for i, story in enumerate(stories):
            if len(stories) > 1:
                print(f"\n--- Story {i+1} ---")
            print(f"\n{args.prompt}{story}")
            print("-" * 60)
        
        print(f"\nGenerated {len(stories)} story(ies) in {generation_time:.2f} seconds")


if __name__ == "__main__":
    import torch.nn.functional as F
    main()
