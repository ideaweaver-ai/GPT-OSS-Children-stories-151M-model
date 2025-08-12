"""
Main Training Script for GPT-OSS Advanced Children's Stories Model
"""

import argparse
import os
import json
import torch
from typing import Dict, Any

from config import ModelConfig, TrainingConfig, DataConfig, get_config_by_name, MODEL_CONFIGS, TRAINING_CONFIGS
from training.advanced_trainer import create_trainer_from_configs
from data.data_processor import AdvancedDataProcessor


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train GPT-OSS Advanced Children's Stories Model")
    
    # Model configuration
    parser.add_argument("--model-size", type=str, default="small", 
                       choices=["tiny", "small", "medium", "large"],
                       help="Predefined model size")
    parser.add_argument("--n-layer", type=int, help="Number of transformer layers")
    parser.add_argument("--n-head", type=int, help="Number of attention heads")
    parser.add_argument("--n-embd", type=int, help="Embedding dimension")
    parser.add_argument("--block-size", type=int, help="Context window size")
    parser.add_argument("--num-experts", type=int, help="Number of MoE experts")
    parser.add_argument("--experts-per-token", type=int, help="Top-k experts per token")
    parser.add_argument("--sliding-window", type=int, help="Sliding window size")
    parser.add_argument("--multi-token-predict", type=int, help="Number of tokens to predict ahead")
    
    # Training configuration
    parser.add_argument("--training-preset", type=str, default="standard",
                       choices=["debug", "fast", "standard", "production"],
                       help="Predefined training configuration")
    parser.add_argument("--learning-rate", type=float, help="Learning rate")
    parser.add_argument("--batch-size", type=int, help="Batch size")
    parser.add_argument("--max-iters", type=int, help="Maximum training iterations")
    parser.add_argument("--eval-interval", type=int, help="Evaluation interval")
    parser.add_argument("--gradient-accumulation-steps", type=int, help="Gradient accumulation steps")
    parser.add_argument("--warmup-steps", type=int, help="Learning rate warmup steps")
    parser.add_argument("--weight-decay", type=float, help="Weight decay")
    parser.add_argument("--max-grad-norm", type=float, help="Gradient clipping norm")
    
    # Data configuration
    parser.add_argument("--dataset-name", type=str, default="ajibawa-2023/Children-Stories-Collection",
                       help="HuggingFace dataset name")
    parser.add_argument("--dataset-path", type=str, help="Path to local dataset")
    parser.add_argument("--max-seq-length", type=int, help="Maximum sequence length")
    parser.add_argument("--data-split-ratios", type=float, nargs=3, 
                       default=[0.8, 0.1, 0.1], help="Train/val/test split ratios")
    
    # Training options
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints",
                       help="Directory to save checkpoints")
    parser.add_argument("--resume-from", type=str, help="Resume training from checkpoint")
    parser.add_argument("--device", type=str, default="auto",
                       choices=["auto", "cuda", "cpu", "mps"],
                       help="Device to use for training")
    parser.add_argument("--compile-model", action="store_true",
                       help="Compile model with PyTorch 2.0")
    parser.add_argument("--mixed-precision", action="store_true", default=True,
                       help="Use mixed precision training")
    parser.add_argument("--use-ddp", action="store_true",
                       help="Use distributed data parallel")
    
    # Logging and monitoring
    parser.add_argument("--use-wandb", action="store_true",
                       help="Use Weights & Biases for logging")
    parser.add_argument("--wandb-project", type=str, default="gpt-oss-children-stories",
                       help="Weights & Biases project name")
    parser.add_argument("--wandb-run-name", type=str, help="Weights & Biases run name")
    parser.add_argument("--log-interval", type=int, default=100,
                       help="Logging interval")
    
    # Data processing
    parser.add_argument("--process-data", action="store_true",
                       help="Process raw data before training")
    parser.add_argument("--data-dir", type=str, default="data",
                       help="Directory for processed data")
    parser.add_argument("--force-reprocess", action="store_true",
                       help="Force reprocessing even if data exists")
    parser.add_argument("--beast-mode", action="store_true",
                       help="Enable aggressive multiprocessing for high-performance systems (H100, etc.)")
    parser.add_argument("--max-workers", type=int, 
                       help="Maximum number of worker processes for data processing")
    
    # Evaluation and generation
    parser.add_argument("--eval-only", action="store_true",
                       help="Only run evaluation, no training")
    parser.add_argument("--generate-samples", action="store_true",
                       help="Generate sample stories after training")
    parser.add_argument("--num-samples", type=int, default=5,
                       help="Number of sample stories to generate")
    
    return parser.parse_args()


def setup_configs(args) -> tuple:
    """Setup configurations from arguments"""
    
    # Get base configurations
    model_config = MODEL_CONFIGS.get(args.model_size, MODEL_CONFIGS['small'])
    training_config = TRAINING_CONFIGS.get(args.training_preset, TRAINING_CONFIGS['standard'])
    data_config = DataConfig()
    
    # Override with command line arguments
    config_overrides = vars(args)
    
    # Update model config
    for key, value in config_overrides.items():
        if value is not None and hasattr(model_config, key):
            setattr(model_config, key, value)
    
    # Update training config
    for key, value in config_overrides.items():
        if value is not None and hasattr(training_config, key):
            setattr(training_config, key, value)
    
    # Update data config
    for key, value in config_overrides.items():
        if value is not None and hasattr(data_config, key):
            setattr(data_config, key, value)
    
    # Handle beast mode
    if args.beast_mode:
        data_config.aggressive_multiprocessing = True
        print("🚀 BEAST MODE ACTIVATED! Using aggressive multiprocessing...")
    
    # Handle manual max_workers override
    if args.max_workers:
        data_config.max_workers = args.max_workers
        print(f"🔧 Manual override: Using {args.max_workers} worker processes")
    
    return model_config, training_config, data_config


def process_data_if_needed(data_config: DataConfig, args) -> bool:
    """Process data if needed"""
    data_dir = args.data_dir
    
    # Check if data already exists
    required_files = ['train.bin', 'val.bin', 'test.bin']
    data_exists = all(os.path.exists(os.path.join(data_dir, f)) for f in required_files)
    
    if data_exists and not args.force_reprocess and not args.process_data:
        print(f"Processed data found in {data_dir}, skipping data processing")
        return True
    
    if args.process_data or args.force_reprocess or not data_exists:
        print("Processing data...")
        processor = AdvancedDataProcessor(data_config)
        
        try:
            results = processor.process_dataset(data_dir)
            print("Data processing completed successfully!")
            print(f"Results: {results}")
            return True
        except Exception as e:
            print(f"Error during data processing: {e}")
            return False
    
    return data_exists


def save_configs(model_config, training_config, data_config, checkpoint_dir):
    """Save configurations to checkpoint directory"""
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    configs = {
        'model_config': model_config.__dict__,
        'training_config': training_config.__dict__,
        'data_config': data_config.__dict__
    }
    
    config_path = os.path.join(checkpoint_dir, 'training_configs.json')
    with open(config_path, 'w') as f:
        json.dump(configs, f, indent=2, default=str)
    
    print(f"Saved training configurations to {config_path}")


def generate_sample_stories(trainer, num_samples=5):
    """Generate sample stories after training"""
    print(f"\nGenerating {num_samples} sample stories...")
    
    from generate import AdvancedTextGenerator
    
    # Save a temporary checkpoint for generation
    temp_checkpoint_path = os.path.join(trainer.checkpoint_dir, 'temp_for_generation.pt')
    trainer.save_checkpoint()
    
    try:
        # Create generator
        generator = AdvancedTextGenerator(
            os.path.join(trainer.checkpoint_dir, 'best_model.pt'),
            device=str(trainer.device)
        )
        
        # Sample prompts
        prompts = [
            "Once upon a time",
            "In a magical forest",
            "There was a brave little mouse",
            "Long ago in a distant kingdom",
            "A young girl discovered"
        ]
        
        print("-" * 60)
        for i, prompt in enumerate(prompts[:num_samples]):
            print(f"\nSample {i+1} - Prompt: '{prompt}'")
            print("-" * 40)
            
            stories = generator.generate(
                prompt=prompt,
                max_new_tokens=150,
                temperature=0.8,
                top_p=0.9
            )
            
            print(f"{prompt}{stories[0]}")
            print("-" * 40)
        
        print("-" * 60)
        
    except Exception as e:
        print(f"Error generating samples: {e}")


def main():
    """Main training function"""
    args = parse_args()
    
    print("="*60)
    print("GPT-OSS Advanced Children's Stories - Training")
    print("="*60)
    
    # Setup configurations
    print("Setting up configurations...")
    model_config, training_config, data_config = setup_configs(args)
    
    print(f"Model configuration:")
    print(f"  Size: {args.model_size}")
    print(f"  Layers: {model_config.n_layer}")
    print(f"  Heads: {model_config.n_head}")
    print(f"  Embedding dim: {model_config.n_embd}")
    print(f"  Context size: {model_config.block_size}")
    print(f"  MoE experts: {model_config.num_experts}")
    print(f"  Multi-token predict: {model_config.multi_token_predict}")
    
    print(f"\nTraining configuration:")
    print(f"  Preset: {args.training_preset}")
    print(f"  Max iterations: {training_config.max_iters}")
    print(f"  Batch size: {training_config.batch_size}")
    print(f"  Learning rate: {training_config.learning_rate}")
    print(f"  Mixed precision: {training_config.use_mixed_precision}")
    print(f"  Model compilation: {training_config.compile_model}")
    
    # Process data if needed
    print(f"\nData processing...")
    if not process_data_if_needed(data_config, args):
        print("Data processing failed. Exiting.")
        return
    
    # Save configurations
    save_configs(model_config, training_config, data_config, args.checkpoint_dir)
    
    # Create trainer
    print("\nInitializing trainer...")
    trainer = create_trainer_from_configs(
        model_config=model_config,
        training_config=training_config,
        data_config=data_config,
        checkpoint_dir=args.checkpoint_dir
    )
    
    # Resume from checkpoint if specified
    if args.resume_from:
        trainer.load_checkpoint(args.resume_from)
    
    # Run evaluation only if requested
    if args.eval_only:
        print("\nRunning evaluation...")
        eval_results = trainer.evaluate()
        print(f"Evaluation results: {eval_results}")
        return
    
    # Start training
    print(f"\nStarting training...")
    print(f"Checkpoint directory: {args.checkpoint_dir}")
    
    try:
        trainer.train()
        
        # Plot training metrics
        print("\nPlotting training metrics...")
        trainer.plot_metrics()
        
        # Generate sample stories if requested
        if args.generate_samples:
            generate_sample_stories(trainer, args.num_samples)
        
        print("\nTraining completed successfully!")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        trainer.save_checkpoint()
        print("Checkpoint saved")
        
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        trainer.save_checkpoint()
        print("Emergency checkpoint saved")
        raise


if __name__ == "__main__":
    main()
