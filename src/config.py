"""
Configuration classes for GPT-OSS Advanced Children's Stories Model
Combines configurations from GPT-OSS, DeepSeek, and Tiny Stories
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import json
import os


@dataclass
class ModelConfig:
    """Model architecture configuration based on GPT-OSS"""
    # Basic model parameters (GPT-OSS specifications)
    vocab_size: int = 201088  # GPT-OSS vocabulary size
    n_layer: int = 12
    n_head: int = 16
    n_embd: int = 768
    block_size: int = 8192  # Extended context length like GPT-OSS
    # Note: GPT-OSS does NOT use dropout - removed for compliance
    bias: bool = True
    
    # Advanced attention features (from GPT-OSS)
    num_key_value_heads: int = 4
    sliding_window: int = 128         # GPT-OSS uses 128 (updated to match official)
    use_attention_sinks: bool = True
    head_dim: int = 48  # Will be computed as n_embd // n_head if not specified
    
    # RoPE configuration (GPT-OSS specifications)
    rope_theta: float = 150000.0      # GPT-OSS uses 150k (vs standard 10k)
    rope_scaling_factor: float = 32.0 # GPT-OSS scaling factor
    rope_ntk_alpha: float = 1.0       # YaRN scaling parameter
    rope_ntk_beta: float = 32.0       # YaRN scaling parameter
    
    # MoE configuration (scaled down from GPT-OSS)
    num_experts: int = 8
    experts_per_token: int = 2
    expert_capacity: float = 1.25
    moe_aux_loss_coeff: float = 0.01
    moe_layers: List[int] = field(default_factory=lambda: [2, 4, 6, 8, 10])  # Which layers use MoE
    
    # Multi-token prediction (GPT-OSS uses 2)
    multi_token_predict: int = 2
    
    # Advanced optimizations
    use_swiglu: bool = True
    use_rmsnorm: bool = True
    swiglu_limit: float = 7.0
    
    # MXFP4 Quantization (GPT-OSS native quantization)
    use_mxfp4_quantization: bool = True   # Enable MXFP4 for MoE layers
    mxfp4_block_size: int = 32           # MXFP4 block size (standard: 32)
    quantize_moe_only: bool = True       # Only quantize MoE layers (GPT-OSS pattern)
    
    # Legacy quantization settings (for compatibility)
    use_quantization: bool = True        # Enabled by default for GPT-OSS compliance
    quantization_bits: int = 4           # 4-bit precision
    
    def __post_init__(self):
        # Compute head_dim if not specified
        if self.head_dim * self.n_head != self.n_embd:
            self.head_dim = self.n_embd // self.n_head
        
        # Ensure consistency
        assert self.n_embd % self.n_head == 0, "n_embd must be divisible by n_head"
        assert self.num_key_value_heads <= self.n_head, "num_key_value_heads must be <= n_head"
        assert self.n_head % self.num_key_value_heads == 0, "n_head must be divisible by num_key_value_heads"


@dataclass
class TrainingConfig:
    """Training configuration with advanced optimizations"""
    # Basic training parameters
    learning_rate: float = 3e-4
    max_iters: int = 50000
    warmup_steps: int = 2000
    min_lr: float = 3e-5
    eval_iters: int = 500
    eval_interval: int = 1000
    batch_size: int = 16
    gradient_accumulation_steps: int = 4
    
    # Optimizer parameters
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    max_grad_norm: float = 1.0
    
    # Learning rate scheduling
    lr_scheduler: str = "cosine"  # "cosine", "linear", "constant"
    lr_decay_iters: int = 50000
    
    # Advanced training features
    use_mixed_precision: bool = True
    compile_model: bool = True  # PyTorch 2.0 compilation
    use_flash_attention: bool = True
    
    # Checkpointing
    save_checkpoint_every: int = 5000
    keep_checkpoints: int = 5
    save_best_model: bool = True
    
    # Data parameters
    data_split_ratios: List[float] = field(default_factory=lambda: [0.8, 0.1, 0.1])  # train, val, test
    
    # Logging and monitoring
    log_interval: int = 100
    use_wandb: bool = True
    wandb_project: str = "gpt-oss-children-stories"
    wandb_run_name: Optional[str] = None
    
    # Hardware optimization
    device: str = "auto"  # "auto", "cuda", "cpu", "mps"
    num_workers: int = 4
    pin_memory: bool = True
    
    # Distributed training
    use_ddp: bool = False
    local_rank: int = -1


@dataclass
class DataConfig:
    """Data processing configuration"""
    # Dataset parameters
    dataset_name: str = "ajibawa-2023/Children-Stories-Collection"
    dataset_path: Optional[str] = None
    text_column: str = "text"
    
    # Processing parameters
    max_seq_length: int = 2048
    stride: int = 1024  # Overlap for sliding window
    min_seq_length: int = 128
    
    # Tokenization (GPT-OSS specific)
    tokenizer_name: str = "o200k_harmony"  # GPT-OSS tokenizer
    add_special_tokens: bool = True
    
    # Data augmentation
    use_data_augmentation: bool = False
    augmentation_prob: float = 0.1
    
    # Preprocessing
    clean_text: bool = True
    remove_duplicates: bool = True
    filter_language: Optional[str] = "en"
    
    # Data splitting
    data_split_ratios: List[float] = field(default_factory=lambda: [0.8, 0.1, 0.1])  # Train/val/test split ratios
    
    # Multiprocessing configuration
    max_workers: Optional[int] = None  # None = auto-detect based on dataset size
    use_multiprocessing: bool = True
    aggressive_multiprocessing: bool = False  # Enable for high-performance systems (H100, etc.)
    memory_efficient_chunking: bool = True  # Use memory-efficient chunking for large datasets


@dataclass
class GenerationConfig:
    """Text generation configuration"""
    # Generation parameters
    max_new_tokens: int = 200
    temperature: float = 0.8
    top_k: Optional[int] = 50
    top_p: Optional[float] = 0.9
    repetition_penalty: float = 1.1
    
    # Sampling strategy
    do_sample: bool = True
    num_beams: int = 1
    early_stopping: bool = True
    
    # Special tokens
    pad_token_id: Optional[int] = None
    eos_token_id: Optional[int] = 50256
    bos_token_id: Optional[int] = 50256
    
    # Output formatting
    skip_special_tokens: bool = True
    clean_up_tokenization_spaces: bool = True


class ConfigManager:
    """Utility class for managing configurations"""
    
    def __init__(self, config_dir: str = "configs"):
        self.config_dir = config_dir
        os.makedirs(config_dir, exist_ok=True)
    
    def save_config(self, config: Any, filename: str) -> None:
        """Save configuration to JSON file"""
        filepath = os.path.join(self.config_dir, filename)
        
        # Convert dataclass to dict
        if hasattr(config, '__dict__'):
            config_dict = config.__dict__
        else:
            config_dict = config
        
        # Handle special types
        config_dict = self._serialize_config(config_dict)
        
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        print(f"Configuration saved to {filepath}")
    
    def load_config(self, config_class: type, filename: str) -> Any:
        """Load configuration from JSON file"""
        filepath = os.path.join(self.config_dir, filename)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Configuration file not found: {filepath}")
        
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        
        # Deserialize special types
        config_dict = self._deserialize_config(config_dict)
        
        return config_class(**config_dict)
    
    def _serialize_config(self, config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Serialize configuration for JSON storage"""
        serialized = {}
        for key, value in config_dict.items():
            if isinstance(value, (list, tuple)):
                serialized[key] = list(value)
            elif hasattr(value, '__dict__'):
                serialized[key] = self._serialize_config(value.__dict__)
            else:
                serialized[key] = value
        return serialized
    
    def _deserialize_config(self, config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Deserialize configuration from JSON"""
        # For now, just return as-is since we're using simple types
        return config_dict
    
    def create_default_configs(self) -> Dict[str, Any]:
        """Create default configuration files"""
        configs = {
            'model': ModelConfig(),
            'training': TrainingConfig(),
            'data': DataConfig(),
            'generation': GenerationConfig()
        }
        
        for name, config in configs.items():
            self.save_config(config, f"{name}_config.json")
        
        return configs


# Predefined model configurations
MODEL_CONFIGS = {
    'tiny': ModelConfig(
        n_layer=6,
        n_head=8,
        n_embd=512,
        block_size=1024,
        num_experts=4,
        moe_layers=[2, 4],
        multi_token_predict=1
    ),
    
    'small': ModelConfig(
        n_layer=12,
        n_head=12,
        n_embd=768,
        block_size=2048,
        num_experts=8,
        moe_layers=[3, 6, 9],
        multi_token_predict=2
    ),
    
    'medium': ModelConfig(
        n_layer=24,
        n_head=16,
        n_embd=1024,
        block_size=4096,
        num_experts=16,
        moe_layers=[4, 8, 12, 16, 20],
        multi_token_predict=2
    ),
    
    'large': ModelConfig(
        n_layer=32,
        n_head=20,
        n_embd=1280,
        block_size=8192,
        num_experts=32,
        num_key_value_heads=5,  # GQA 4:1 ratio
        moe_layers=[4, 8, 12, 16, 20, 24, 28],
        multi_token_predict=2,
        sliding_window=512,
        # Optimized for 201K vocabulary
        use_mxfp4_quantization=True,
        use_rmsnorm=True,
        use_swiglu=True,
    ),
    
    'xlarge': ModelConfig(
        n_layer=48,
        n_head=32,
        n_embd=1536,
        block_size=8192,
        num_experts=64,
        num_key_value_heads=8,  # GQA 4:1 ratio
        moe_layers=[6, 12, 18, 24, 30, 36, 42],
        multi_token_predict=3,
        sliding_window=1024,
        # High-end GPU configuration
        use_mxfp4_quantization=True,
        use_rmsnorm=True,
        use_swiglu=True,
    )
}

# Predefined training configurations
TRAINING_CONFIGS = {
    'debug': TrainingConfig(
        max_iters=1000,
        eval_interval=100,
        batch_size=4,
        use_wandb=False
    ),
    
    'fast': TrainingConfig(
        max_iters=10000,
        eval_interval=500,
        batch_size=8,
        gradient_accumulation_steps=2
    ),
    
    'standard': TrainingConfig(
        max_iters=50000,
        eval_interval=1000,
        batch_size=16,
        gradient_accumulation_steps=4
    ),
    
    'production': TrainingConfig(
        max_iters=100000,
        eval_interval=2000,
        batch_size=32,
        gradient_accumulation_steps=8,
        use_ddp=True
    ),
    
    # Extended training for high-end GPU
    'extended': TrainingConfig(
        max_iters=25000,
        eval_interval=1000,
        batch_size=16,
        gradient_accumulation_steps=8,  # Effective batch size: 128
        learning_rate=2e-4,
        warmup_steps=2500,
        use_mixed_precision=True,
        compile_model=True,
        save_checkpoint_every=2500,
        use_wandb=True
    ),
    
    # Long training for large models
    'marathon': TrainingConfig(
        max_iters=50000,
        eval_interval=2000,
        batch_size=24,
        gradient_accumulation_steps=8,  # Effective batch size: 192
        learning_rate=1.5e-4,
        warmup_steps=5000,
        use_mixed_precision=True,
        compile_model=True,
        save_checkpoint_every=5000,
        use_wandb=True
    )
}


def get_config(config_type: str, config_name: str) -> Any:
    """Get predefined configuration"""
    if config_type == 'model':
        return MODEL_CONFIGS.get(config_name, ModelConfig())
    elif config_type == 'training':
        return TRAINING_CONFIGS.get(config_name, TrainingConfig())
    else:
        raise ValueError(f"Unknown config type: {config_type}")


def create_config_from_args(args: Dict[str, Any]) -> tuple:
    """Create configurations from command line arguments"""
    # Extract model config args
    model_args = {k: v for k, v in args.items() if k in ModelConfig.__annotations__}
    model_config = ModelConfig(**model_args)
    
    # Extract training config args
    training_args = {k: v for k, v in args.items() if k in TrainingConfig.__annotations__}
    training_config = TrainingConfig(**training_args)
    
    # Extract data config args
    data_args = {k: v for k, v in args.items() if k in DataConfig.__annotations__}
    data_config = DataConfig(**data_args)
    
    return model_config, training_config, data_config


def get_config_by_name(model_name: str = 'small', training_name: str = 'standard') -> tuple:
    """Get configurations by name"""
    model_config = MODEL_CONFIGS.get(model_name)
    training_config = TRAINING_CONFIGS.get(training_name)
    
    if model_config is None:
        raise ValueError(f"Unknown model config: {model_name}. Available: {list(MODEL_CONFIGS.keys())}")
    if training_config is None:
        raise ValueError(f"Unknown training config: {training_name}. Available: {list(TRAINING_CONFIGS.keys())}")
    
    return model_config, training_config


if __name__ == "__main__":
    # Create default configuration files
    config_manager = ConfigManager()
    configs = config_manager.create_default_configs()
    
    print("Created default configuration files:")
    for name in configs.keys():
        print(f"  - {name}_config.json")
    
    print("\nAvailable model configurations:")
    for name in MODEL_CONFIGS.keys():
        print(f"  - {name}")
    
    print("\nAvailable training configurations:")
    for name in TRAINING_CONFIGS.keys():
        print(f"  - {name}")
