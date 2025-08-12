"""
Advanced Data Processing Pipeline for GPT-OSS Children's Stories
Combines data processing approaches from all three models
"""

import os
import json
import numpy as np
import tiktoken
from typing import List, Dict, Tuple, Optional, Union
from datasets import load_dataset, Dataset
from transformers import GPT2TokenizerFast
import re
from tqdm.auto import tqdm
import multiprocessing as mp
from functools import partial
import pickle
import hashlib

from config import DataConfig


# Global worker variables for multiprocessing
_worker_tokenizer = None
_worker_config = None


def _init_worker(tokenizer_name: str, add_special_tokens: bool, max_seq_length: int):
    """Initialize tokenizer in each worker process"""
    global _worker_tokenizer, _worker_config
    
    # Create a simple config object for the worker
    class WorkerConfig:
        def __init__(self, tokenizer_name, add_special_tokens, max_seq_length):
            self.tokenizer_name = tokenizer_name
            self.add_special_tokens = add_special_tokens
            self.max_seq_length = max_seq_length
    
    _worker_config = WorkerConfig(tokenizer_name, add_special_tokens, max_seq_length)
    
    # Initialize tokenizer in worker - simplified to avoid infinite loops
    try:
        if tokenizer_name == "o200k_harmony":
            # Try openai_harmony first, but fall back quickly if it fails
            try:
                import openai_harmony
                if hasattr(openai_harmony, 'load_harmony_encoding') and hasattr(openai_harmony, 'HarmonyEncodingName'):
                    from openai_harmony import load_harmony_encoding, HarmonyEncodingName
                    _worker_tokenizer = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
                else:
                    raise AttributeError("Required harmony encoding not found")
            except Exception:
                # Quick fallback without printing debug info to avoid spam
                import tiktoken
                _worker_tokenizer = tiktoken.get_encoding("o200k_base")
        elif tokenizer_name == "gpt2":
            # Use tiktoken for GPT-2 to avoid HuggingFace overhead in workers
            import tiktoken
            _worker_tokenizer = tiktoken.get_encoding("gpt2")
        else:
            import tiktoken
            _worker_tokenizer = tiktoken.get_encoding(tokenizer_name)
    except Exception as e:
        # Final fallback - use a simple tiktoken encoder
        import tiktoken
        _worker_tokenizer = tiktoken.get_encoding("gpt2")
        print(f"Worker tokenizer fallback: {e}")


def _process_text_chunk_worker(texts: List[str]) -> List[List[int]]:
    """Worker function to process text chunks"""
    global _worker_tokenizer, _worker_config
    
    sequences = []
    for text in texts:
        # Clean and validate text
        text = _clean_text_worker(text)
        if not _is_valid_story_worker(text):
            continue
            
        # Tokenize text
        tokens = _tokenize_text_worker(text)
        if not tokens or len(tokens) < 10:  # Skip very short sequences
            continue
            
        # Split into sequences
        max_len = _worker_config.max_seq_length
        for i in range(0, len(tokens), max_len):
            sequence = tokens[i:i + max_len]
            if len(sequence) >= 10:  # Minimum sequence length
                sequences.append(sequence)
    
    return sequences


def _clean_text_worker(text: str) -> str:
    """Clean text in worker process"""
    if not isinstance(text, str):
        return ""
    
    # Remove extra whitespace and normalize
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Fix common punctuation issues
    text = re.sub(r'\s+([,.!?;:])', r'\1', text)
    text = re.sub(r'([.!?])\s*([A-Z])', r'\1 \2', text)
    
    # Remove or fix quotes
    text = re.sub(r"[''']", "'", text)
    text = re.sub(r'["""]', '"', text)
    
    return text


def _is_valid_story_worker(text: str) -> bool:
    """Check if text is a valid story in worker process"""
    if len(text) < 50:  # Too short
        return False
    
    if len(text) > 50000:  # Too long
        return False
    
    # Check for story elements
    story_indicators = ['once', 'there', 'was', 'said', 'went', 'came', 'saw', 'found']
    text_lower = text.lower()
    has_story_elements = any(word in text_lower for word in story_indicators)
    
    # Check for inappropriate content
    inappropriate_words = ['violence', 'death', 'kill', 'murder', 'blood', 'war', 'adult', 'mature', 'explicit']
    has_inappropriate = any(word in text_lower for word in inappropriate_words)
    
    return has_story_elements and not has_inappropriate


def _tokenize_text_worker(text: str) -> List[int]:
    """Tokenize text in worker process"""
    global _worker_tokenizer, _worker_config
    
    if hasattr(_worker_tokenizer, 'encode'):
        # tiktoken or harmony tokenizer
        if _worker_config.add_special_tokens:
            # Add BOS token manually
            try:
                tokens = [_worker_tokenizer.encode("<|endoftext|>", disallowed_special=())[0]] + _worker_tokenizer.encode(text, disallowed_special=())
            except:
                tokens = _worker_tokenizer.encode(text, disallowed_special=())
        else:
            tokens = _worker_tokenizer.encode(text, disallowed_special=())
    else:
        # HuggingFace tokenizer
        tokens = _worker_tokenizer.encode(
            text,
            add_special_tokens=_worker_config.add_special_tokens,
            truncation=True,
            max_length=_worker_config.max_seq_length
        )
    
    return tokens


class AdvancedDataProcessor:
    """Advanced data processor with multiple tokenization and processing strategies"""
    
    def __init__(self, config: DataConfig):
        self.config = config
        self.tokenizer = self._setup_tokenizer()
        # Get vocabulary size from tokenizer
        if hasattr(self.tokenizer, 'n_vocab'):
            self.vocab_size = self.tokenizer.n_vocab
        elif hasattr(self.tokenizer, 'vocab_size'):
            self.vocab_size = self.tokenizer.vocab_size
        elif hasattr(self.tokenizer, '__len__'):
            self.vocab_size = len(self.tokenizer)
        else:
            # Default GPT-OSS vocabulary size
            self.vocab_size = 201088
        
        print(f"Data processor initialized:")
        print(f"  Tokenizer: {config.tokenizer_name}")
        print(f"  Vocab size: {self.vocab_size}")
        print(f"  Max sequence length: {config.max_seq_length}")
    
    def _setup_tokenizer(self):
        """Setup tokenizer based on configuration"""
        if self.config.tokenizer_name == "o200k_harmony":
            # Use GPT-OSS harmony tokenizer
            try:
                # Try to use the harmony tokenizer from openai-harmony package
                import openai_harmony
                
                # Check for the correct harmony encoding approach
                if hasattr(openai_harmony, 'load_harmony_encoding') and hasattr(openai_harmony, 'HarmonyEncodingName'):
                    from openai_harmony import load_harmony_encoding, HarmonyEncodingName
                    tokenizer = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
                    print("Using GPT-OSS harmony tokenizer (load_harmony_encoding)")
                    return tokenizer
                else:
                    print("openai-harmony installed but required encoding not found, falling back to o200k_base")
                    tokenizer = tiktoken.get_encoding("o200k_base")
                    print("Using tiktoken o200k_base tokenizer (GPT-OSS compatible)")
                    return tokenizer
            except Exception as e:
                print(f"Failed to load GPT-OSS tokenizer: {e}")
                # Fallback to o200k_base instead of gpt2 for better compatibility
                tokenizer = tiktoken.get_encoding("o200k_base")
                print("Using tiktoken o200k_base tokenizer (fallback)")
                return tokenizer
        elif self.config.tokenizer_name == "gpt2":
            # Legacy GPT-2 support
            try:
                tokenizer = tiktoken.get_encoding("gpt2")
                print("Using tiktoken GPT-2 tokenizer")
                return tokenizer
            except Exception as e:
                print(f"Failed to load tiktoken: {e}")
                # Fallback to HuggingFace tokenizer
                tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                print("Using HuggingFace GPT-2 tokenizer")
                return tokenizer
        else:
            # Use HuggingFace tokenizer for other models
            tokenizer = GPT2TokenizerFast.from_pretrained(self.config.tokenizer_name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            return tokenizer
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        if not self.config.clean_text:
            return text
        
        # Basic cleaning
        text = text.strip()
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\'\"\-\(\)]', '', text)
        
        # Normalize quotes
        text = re.sub(r'["""]', '"', text)
        text = re.sub(r"[''']", "'", text)
        
        # Remove multiple consecutive punctuation
        text = re.sub(r'([.!?]){2,}', r'\1', text)
        
        # Ensure sentences end with proper punctuation
        sentences = text.split('.')
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and not sentence.endswith(('.', '!', '?')):
                sentence += '.'
            if sentence:
                cleaned_sentences.append(sentence)
        
        return ' '.join(cleaned_sentences)
    
    def is_valid_story(self, text: str) -> bool:
        """Check if text is a valid children's story"""
        if len(text) < self.config.min_seq_length:
            return False
        
        # Check for minimum word count
        words = text.split()
        if len(words) < 20:  # Minimum 20 words
            return False
        
        # Check for story-like content (basic heuristics)
        story_indicators = [
            'once upon a time', 'there was', 'there were', 'long ago',
            'in a', 'story', 'tale', 'adventure', 'little', 'big'
        ]
        
        text_lower = text.lower()
        has_story_elements = any(indicator in text_lower for indicator in story_indicators)
        
        # Check for appropriate content (avoid adult content)
        inappropriate_words = [
            'violence', 'death', 'kill', 'murder', 'blood', 'war',
            'adult', 'mature', 'explicit'
        ]
        
        has_inappropriate = any(word in text_lower for word in inappropriate_words)
        
        return has_story_elements and not has_inappropriate
    
    def tokenize_text(self, text: str) -> List[int]:
        """Tokenize text using the configured tokenizer"""
        if hasattr(self.tokenizer, 'encode'):
            # tiktoken or similar
            if self.config.add_special_tokens:
                # Add BOS token manually for tiktoken
                tokens = [self.tokenizer.encode("<|endoftext|>", disallowed_special=())[0]] + self.tokenizer.encode(text, disallowed_special=())
            else:
                tokens = self.tokenizer.encode(text, disallowed_special=())
        else:
            # HuggingFace tokenizer
            tokens = self.tokenizer.encode(
                text,
                add_special_tokens=self.config.add_special_tokens,
                truncation=True,
                max_length=self.config.max_seq_length
            )
        
        return tokens
    
    def process_single_text(self, text: str) -> List[List[int]]:
        """Process a single text into tokenized sequences"""
        # Clean text
        cleaned_text = self.clean_text(text)
        
        # Validate story
        if not self.is_valid_story(cleaned_text):
            return []
        
        # Tokenize
        tokens = self.tokenize_text(cleaned_text)
        
        # Split into chunks with sliding window
        sequences = []
        max_len = self.config.max_seq_length
        stride = self.config.stride
        
        for i in range(0, len(tokens) - max_len + 1, stride):
            chunk = tokens[i:i + max_len]
            if len(chunk) == max_len:  # Only keep full-length sequences
                sequences.append(chunk)
        
        return sequences
    
    def load_dataset_from_hf(self) -> Dataset:
        """Load dataset from Hugging Face"""
        print(f"Loading dataset: {self.config.dataset_name}")
        
        try:
            dataset = load_dataset(self.config.dataset_name)
            
            # Get the appropriate split
            if 'train' in dataset:
                data = dataset['train']
            else:
                # Use the first available split
                split_name = list(dataset.keys())[0]
                data = dataset[split_name]
                print(f"Using split: {split_name}")
            
            print(f"Loaded {len(data)} examples")
            return data
        
        except Exception as e:
            print(f"Error loading dataset: {e}")
            raise
    
    def load_dataset_from_file(self) -> List[str]:
        """Load dataset from local file"""
        if not os.path.exists(self.config.dataset_path):
            raise FileNotFoundError(f"Dataset file not found: {self.config.dataset_path}")
        
        texts = []
        
        if self.config.dataset_path.endswith('.json'):
            with open(self.config.dataset_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict) and self.config.text_column in item:
                            texts.append(item[self.config.text_column])
                        elif isinstance(item, str):
                            texts.append(item)
                elif isinstance(data, dict) and self.config.text_column in data:
                    texts = data[self.config.text_column]
        
        elif self.config.dataset_path.endswith('.txt'):
            with open(self.config.dataset_path, 'r', encoding='utf-8') as f:
                content = f.read()
                # Split by double newlines (assuming stories are separated this way)
                texts = [text.strip() for text in content.split('\n\n') if text.strip()]
        
        print(f"Loaded {len(texts)} texts from file")
        return texts
    
    def parallel_process_texts(self, texts: List[str], num_processes: Optional[int] = None) -> List[List[int]]:
        """Process texts in parallel"""
        if num_processes is None:
            # Check if config has max_workers setting
            if hasattr(self.config, 'max_workers') and self.config.max_workers is not None:
                num_processes = self.config.max_workers
                print(f"Using configured max_workers: {num_processes}")
            else:
                # Intelligent process count based on system resources and data size
                cpu_count = mp.cpu_count()
                
                # Check if aggressive multiprocessing is enabled
                aggressive = getattr(self.config, 'aggressive_multiprocessing', False)
                
                # For high-core-count systems (like H100 setups), be more aggressive
                if cpu_count >= 100:  # High-performance systems (H100, etc.)
                    if aggressive:
                        # BEAST MODE: Use most of your cores for massive datasets
                        if len(texts) > 500000:  # Massive datasets
                            num_processes = min(cpu_count * 3 // 4, 128)  # Use 75% of cores, max 128
                        elif len(texts) > 100000:  # Large datasets  
                            num_processes = min(cpu_count * 2 // 3, 96)   # Use 66% of cores, max 96
                        elif len(texts) > 10000:  # Medium datasets
                            num_processes = min(cpu_count // 2, 64)      # Use 50% of cores, max 64
                        else:  # Small datasets
                            num_processes = min(cpu_count // 4, 32)      # Use 25% of cores, max 32
                    else:
                        # Conservative high-performance mode
                        if len(texts) > 500000:  # Massive datasets
                            num_processes = min(cpu_count // 2, 64)  # Use up to half cores, max 64
                        elif len(texts) > 100000:  # Large datasets  
                            num_processes = min(cpu_count // 3, 48)  # Use 1/3 cores, max 48
                        elif len(texts) > 10000:  # Medium datasets
                            num_processes = min(cpu_count // 4, 32)  # Use 1/4 cores, max 32
                        else:  # Small datasets
                            num_processes = min(cpu_count // 8, 16)  # Use 1/8 cores, max 16
                else:
                    # Original logic for normal systems
                    if len(texts) > 100000:  # Large dataset
                        num_processes = min(cpu_count, 8)  # Up to 8 processes for large datasets
                    elif len(texts) > 10000:  # Medium dataset
                        num_processes = min(cpu_count, 6)  # Up to 6 processes for medium datasets
                    else:  # Small dataset
                        num_processes = min(cpu_count, 4)  # Up to 4 processes for small datasets
                
                # Always leave at least a few cores free for system (more for high-core systems)
                cores_to_reserve = max(4, cpu_count // 20)  # Reserve 4 cores minimum, or 5% of total
                num_processes = max(1, min(num_processes, cpu_count - cores_to_reserve))
                
                print(f"🚀 Auto-detected {num_processes} processes (CPU cores: {cpu_count}, dataset size: {len(texts):,}, reserved cores: {cores_to_reserve})")
        
        print(f"Processing {len(texts)} texts using {num_processes} processes...")
        
        # Split texts into chunks for each process
        chunk_size = max(1, len(texts) // num_processes)
        text_chunks = [texts[i:i + chunk_size] for i in range(0, len(texts), chunk_size)]
        
        # Remove empty chunks
        text_chunks = [chunk for chunk in text_chunks if chunk]
        
        # Use multiprocessing with timeout to prevent infinite loops
        try:
            print(f"Using multiprocessing with {len(text_chunks)} processes...")
            
            # Set a reasonable timeout based on data size and system capacity
            # High-performance systems can process faster, so scale timeout accordingly
            cpu_count = mp.cpu_count()
            if cpu_count >= 100:  # High-performance systems
                timeout_seconds = max(600, len(texts) * 0.05)  # At least 10 minutes, but less per text
            else:
                timeout_seconds = max(300, len(texts) * 0.1)  # At least 5 minutes, more for large datasets
            
            with mp.Pool(processes=len(text_chunks), 
                        initializer=_init_worker, 
                        initargs=(self.config.tokenizer_name, self.config.add_special_tokens, self.config.max_seq_length)) as pool:
                
                # Use map_async with timeout to prevent hanging
                async_result = pool.map_async(_process_text_chunk_worker, text_chunks)
                results = async_result.get(timeout=timeout_seconds)
                
            print("✅ Multiprocessing completed successfully!")
        except mp.TimeoutError:
            print(f"⚠️  Multiprocessing timed out after {timeout_seconds}s, falling back to single-threaded...")
            results = [self._process_text_chunk(chunk) for chunk in text_chunks]
        except Exception as e:
            print(f"⚠️  Multiprocessing failed ({e}), falling back to single-threaded...")
            results = [self._process_text_chunk(chunk) for chunk in text_chunks]
        
        # Flatten results
        all_sequences = []
        for chunk_sequences in results:
            if chunk_sequences:  # Check if chunk_sequences is not None/empty
                all_sequences.extend(chunk_sequences)
        
        print(f"Generated {len(all_sequences)} sequences")
        return all_sequences
    
    def _process_text_chunk(self, texts: List[str]) -> List[List[int]]:
        """Process a chunk of texts (for multiprocessing)"""
        # Re-initialize tokenizer in each process to avoid pickling issues
        if not hasattr(self, 'tokenizer') or self.tokenizer is None:
            self._setup_tokenizer()
            
        sequences = []
        for text in texts:
            text_sequences = self.process_single_text(text)
            sequences.extend(text_sequences)
        return sequences
    
    def remove_duplicates(self, sequences: List[List[int]]) -> List[List[int]]:
        """Remove duplicate sequences"""
        if not self.config.remove_duplicates:
            return sequences
        
        print("Removing duplicates...")
        
        # Create hashes for deduplication
        sequence_hashes = set()
        unique_sequences = []
        
        for seq in tqdm(sequences, desc="Deduplicating"):
            # Create hash of the sequence
            seq_hash = hashlib.md5(str(seq).encode()).hexdigest()
            
            if seq_hash not in sequence_hashes:
                sequence_hashes.add(seq_hash)
                unique_sequences.append(seq)
        
        print(f"Removed {len(sequences) - len(unique_sequences)} duplicates")
        return unique_sequences
    
    def split_data(self, sequences: List[List[int]]) -> Tuple[List[List[int]], List[List[int]], List[List[int]]]:
        """Split data into train/validation/test sets"""
        total_sequences = len(sequences)
        
        # Calculate split sizes
        if not hasattr(self.config, 'data_split_ratios'):
            print(f"Warning: data_split_ratios not found in config. Available attributes: {dir(self.config)}")
            # Use default values
            train_ratio, val_ratio, test_ratio = 0.8, 0.1, 0.1
        else:
            train_ratio, val_ratio, test_ratio = self.config.data_split_ratios
        train_size = int(total_sequences * train_ratio)
        val_size = int(total_sequences * val_ratio)
        
        # Shuffle sequences
        np.random.seed(42)  # For reproducibility
        indices = np.random.permutation(total_sequences)
        
        # Split indices
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]
        
        # Create splits
        train_data = [sequences[i] for i in train_indices]
        val_data = [sequences[i] for i in val_indices]
        test_data = [sequences[i] for i in test_indices]
        
        print(f"Data split:")
        print(f"  Train: {len(train_data):,} sequences")
        print(f"  Validation: {len(val_data):,} sequences")
        print(f"  Test: {len(test_data):,} sequences")
        
        return train_data, val_data, test_data
    
    def save_binary_data(self, sequences: List[List[int]], filename: str, output_dir: str = 'data'):
        """Save tokenized sequences as binary file"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Flatten sequences into single array
        all_tokens = []
        for seq in sequences:
            all_tokens.extend(seq)
        
        # Convert to numpy array
        # Use uint32 for GPT-OSS tokenizer (vocab_size = 201,088 > uint16 max of 65,535)
        token_array = np.array(all_tokens, dtype=np.uint32)
        
        # Save as binary file
        output_path = os.path.join(output_dir, filename)
        token_array.tofile(output_path)
        
        print(f"Saved {len(token_array):,} tokens to {output_path}")
        
        # Also save metadata
        metadata = {
            'num_sequences': len(sequences),
            'total_tokens': len(token_array),
            'vocab_size': self.vocab_size,
            'max_seq_length': self.config.max_seq_length,
            'tokenizer': self.config.tokenizer_name
        }
        
        metadata_path = os.path.join(output_dir, filename.replace('.bin', '_metadata.json'))
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def process_dataset(self, output_dir: str = 'data', use_multiprocessing: Optional[bool] = None):
        """Main processing pipeline"""
        print("Starting data processing pipeline...")
        
        # Load raw data
        if self.config.dataset_path:
            texts = self.load_dataset_from_file()
        else:
            dataset = self.load_dataset_from_hf()
            texts = [item[self.config.text_column] for item in dataset]
        
        # Determine multiprocessing setting
        if use_multiprocessing is None:
            use_multiprocessing = getattr(self.config, 'use_multiprocessing', True)
        
        # Process texts into sequences
        if use_multiprocessing:
            sequences = self.parallel_process_texts(texts)
        else:
            print("Using single-threaded processing (multiprocessing disabled)")
            sequences = []
            for text in tqdm(texts, desc="Processing texts"):
                text_sequences = self.process_single_text(text)
                sequences.extend(text_sequences)
        
        if not sequences:
            raise ValueError("No valid sequences generated from the dataset")
        
        # Remove duplicates
        sequences = self.remove_duplicates(sequences)
        
        # Split data
        train_data, val_data, test_data = self.split_data(sequences)
        
        # Save binary files
        self.save_binary_data(train_data, 'train.bin', output_dir)
        self.save_binary_data(val_data, 'val.bin', output_dir)
        self.save_binary_data(test_data, 'test.bin', output_dir)
        
        # Save processing config
        config_path = os.path.join(output_dir, 'processing_config.json')
        with open(config_path, 'w') as f:
            config_dict = {
                'dataset_name': self.config.dataset_name,
                'dataset_path': self.config.dataset_path,
                'tokenizer_name': self.config.tokenizer_name,
                'vocab_size': self.vocab_size,
                'max_seq_length': self.config.max_seq_length,
                'stride': self.config.stride,
                'data_split_ratios': self.config.data_split_ratios,
                'total_sequences': len(sequences),
                'train_sequences': len(train_data),
                'val_sequences': len(val_data),
                'test_sequences': len(test_data)
            }
            json.dump(config_dict, f, indent=2)
        
        print("Data processing completed successfully!")
        print(f"Output directory: {output_dir}")
        return {
            'train_sequences': len(train_data),
            'val_sequences': len(val_data),
            'test_sequences': len(test_data),
            'total_tokens': sum(len(seq) for seq in sequences),
            'vocab_size': self.vocab_size
        }


class DatasetValidator:
    """Utility class for validating processed datasets"""
    
    @staticmethod
    def validate_binary_file(filepath: str, expected_dtype: np.dtype = np.uint32):
        """Validate a binary data file"""
        if not os.path.exists(filepath):
            return False, f"File not found: {filepath}"
        
        try:
            data = np.fromfile(filepath, dtype=expected_dtype)
            
            # Basic checks
            if len(data) == 0:
                return False, "File is empty"
            
            # Check for reasonable token values
            max_token = np.max(data)
            min_token = np.min(data)
            
            if min_token < 0:
                return False, f"Negative token values found: {min_token}"
            
            if max_token > 100000:  # Reasonable upper bound for vocab
                return False, f"Suspiciously high token values: {max_token}"
            
            return True, f"Valid: {len(data):,} tokens, range [{min_token}, {max_token}]"
        
        except Exception as e:
            return False, f"Error reading file: {e}"
    
    @staticmethod
    def validate_dataset_directory(data_dir: str):
        """Validate all files in a dataset directory"""
        required_files = ['train.bin', 'val.bin', 'test.bin']
        results = {}
        
        for filename in required_files:
            filepath = os.path.join(data_dir, filename)
            is_valid, message = DatasetValidator.validate_binary_file(filepath)
            results[filename] = {'valid': is_valid, 'message': message}
        
        # Check metadata files
        metadata_files = ['train_metadata.json', 'val_metadata.json', 'test_metadata.json']
        for filename in metadata_files:
            filepath = os.path.join(data_dir, filename)
            if os.path.exists(filepath):
                try:
                    with open(filepath, 'r') as f:
                        json.load(f)
                    results[filename] = {'valid': True, 'message': 'Valid JSON'}
                except Exception as e:
                    results[filename] = {'valid': False, 'message': f'Invalid JSON: {e}'}
            else:
                results[filename] = {'valid': False, 'message': 'File not found'}
        
        return results


def create_high_performance_config() -> DataConfig:
    """Create optimized configuration for high-performance systems (H100, etc.)"""
    from config import DataConfig
    import multiprocessing as mp
    
    cpu_count = mp.cpu_count()
    print(f"🚀 Detected {cpu_count} CPU cores - configuring for HIGH PERFORMANCE!")
    
    config = DataConfig(
        dataset_name="ajibawa-2023/Children-Stories-Collection",
        max_seq_length=1024,
        stride=512,
        data_split_ratios=[0.8, 0.1, 0.1],
        clean_text=True,
        remove_duplicates=True,
        # HIGH PERFORMANCE SETTINGS
        aggressive_multiprocessing=True,  # BEAST MODE!
        memory_efficient_chunking=True,
        use_multiprocessing=True,
        # For your 208-core system with 896K texts, this will use ~96 cores
        max_workers=None  # Auto-detect for aggressive mode
    )
    
    return config


def main():
    """Example usage of the data processor"""
    from config import DataConfig
    
    # Create high-performance configuration for your beast system
    config = create_high_performance_config()
    
    # Create processor
    processor = AdvancedDataProcessor(config)
    
    # Process dataset with fallback option
    try:
        # Try with multiprocessing first
        results = processor.process_dataset(use_multiprocessing=True)
    except Exception as e:
        print(f"Multiprocessing failed: {e}")
        print("Falling back to single-threaded processing...")
        results = processor.process_dataset(use_multiprocessing=False)
    
    print("\nProcessing Results:")
    for key, value in results.items():
        print(f"  {key}: {value:,}")
    
    # Validate results
    print("\nValidating processed data...")
    validation_results = DatasetValidator.validate_dataset_directory('data')
    
    for filename, result in validation_results.items():
        status = "✓" if result['valid'] else "✗"
        print(f"  {status} {filename}: {result['message']}")


if __name__ == "__main__":
    main()
