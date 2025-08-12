"""
Advanced Trainer for GPT-OSS Children's Stories Model
Combines training optimizations from GPT-OSS, DeepSeek, and Tiny Stories
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
import numpy as np
import os
import json
import time
import math
import gc
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import asdict
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import psutil

# Distributed training imports
try:
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.distributed import init_process_group, destroy_process_group
    import torch.distributed as dist
    DDP_AVAILABLE = True
except ImportError:
    DDP_AVAILABLE = False

# Weights & Biases for logging
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from config import ModelConfig, TrainingConfig, DataConfig
from model.gpt_oss_advanced import GPTOSSAdvanced, GPTOSSAdvancedConfig


class ChildrenStoriesDataset(Dataset):
    """Dataset class for children's stories"""
    
    def __init__(self, data_path: str, block_size: int, split: str = 'train'):
        self.block_size = block_size
        self.split = split
        
        # Load data
        data_file = os.path.join(data_path, f'{split}.bin')
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"Data file not found: {data_file}")
        
        # Memory-mapped file for efficient loading
        self.data = np.memmap(data_file, dtype=np.uint16, mode='r')
        print(f"Loaded {split} dataset: {len(self.data):,} tokens")
    
    def __len__(self):
        return len(self.data) - self.block_size
    
    def __getitem__(self, idx):
        # Get sequence of length block_size + 1
        chunk = torch.from_numpy(self.data[idx:idx + self.block_size + 1].astype(np.int64))
        x = chunk[:-1]  # Input sequence
        y = chunk[1:]   # Target sequence (shifted by 1)
        return x, y


class AdvancedTrainer:
    """Advanced trainer with state-of-the-art optimizations"""
    
    def __init__(
        self, 
        model: GPTOSSAdvanced, 
        train_config: TrainingConfig,
        model_config: ModelConfig,
        data_config: DataConfig,
        checkpoint_dir: str = 'checkpoints'
    ):
        self.model = model
        self.train_config = train_config
        self.model_config = model_config
        self.data_config = data_config
        self.checkpoint_dir = checkpoint_dir
        
        # Create checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Setup device
        self.device = self._setup_device()
        self.model = self.model.to(self.device)
        
        # Setup distributed training if available
        self.is_distributed = self._setup_distributed()
        
        # Wrap model for distributed training
        if self.is_distributed:
            self.model = DDP(self.model, device_ids=[self.device])
            self.raw_model = self.model.module
        else:
            self.raw_model = self.model
        
        # Compile model for PyTorch 2.0+ optimization
        if train_config.compile_model and hasattr(torch, 'compile'):
            print("Compiling model with PyTorch 2.0...")
            self.model = torch.compile(self.model)
        
        # Setup optimizer and scheduler
        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler()
        
        # Setup gradient scaler for mixed precision
        # Use appropriate scaler based on device
        if train_config.use_mixed_precision:
            if self.device.type == 'cuda':
                self.scaler = torch.cuda.amp.GradScaler()
            else:
                # For MPS or CPU, disable grad scaling
                self.scaler = None
        else:
            self.scaler = None
        
        # Training state
        self.current_iter = 0
        self.best_val_loss = float('inf')
        self.metrics = {
            'train_loss': [],
            'val_loss': [],
            'learning_rates': [],
            'grad_norm': [],
            'memory_usage': [],
            'moe_aux_loss': [],
            'tokens_per_sec': []
        }
        
        # Setup logging
        self.use_wandb = train_config.use_wandb and WANDB_AVAILABLE
        if self.use_wandb and (not self.is_distributed or dist.get_rank() == 0):
            self._setup_wandb()
        
        # Load datasets
        self.train_dataset = None
        self.val_dataset = None
        self._setup_datasets()
        
        print(f"Trainer initialized:")
        print(f"  Device: {self.device}")
        print(f"  Model parameters: {self.raw_model.get_num_params():,}")
        print(f"  Distributed: {self.is_distributed}")
        print(f"  Mixed precision: {train_config.use_mixed_precision}")
        print(f"  Model compilation: {train_config.compile_model}")
    
    def _setup_device(self) -> torch.device:
        """Setup compute device"""
        device = getattr(self.train_config, 'device', 'auto')
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
        if device.type == "cuda":
            print(f"  GPU: {torch.cuda.get_device_name()}")
            print(f"  Memory: {torch.cuda.get_device_properties(device).total_memory / 1e9:.1f} GB")
        
        return device
    
    def _setup_distributed(self) -> bool:
        """Setup distributed training"""
        if not self.train_config.use_ddp or not DDP_AVAILABLE:
            return False
        
        if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
            rank = int(os.environ['RANK'])
            world_size = int(os.environ['WORLD_SIZE'])
            local_rank = int(os.environ.get('LOCAL_RANK', 0))
            
            init_process_group(backend='nccl')
            torch.cuda.set_device(local_rank)
            self.device = torch.device(f'cuda:{local_rank}')
            
            print(f"Distributed training: rank {rank}/{world_size}")
            return True
        
        return False
    
    def _setup_optimizer(self) -> torch.optim.Optimizer:
        """Setup optimizer with weight decay"""
        # Separate parameters for weight decay
        decay_params = []
        no_decay_params = []
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                # No weight decay for biases and layer norms
                if 'bias' in name or 'norm' in name or 'embedding' in name:
                    no_decay_params.append(param)
                else:
                    decay_params.append(param)
        
        optimizer_grouped_parameters = [
            {'params': decay_params, 'weight_decay': self.train_config.weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0}
        ]
        
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.train_config.learning_rate,
            betas=(self.train_config.beta1, self.train_config.beta2),
            eps=getattr(self.train_config, 'eps', 1e-8)
        )
        
        print(f"Optimizer: AdamW with {len(decay_params)} decay params, {len(no_decay_params)} no-decay params")
        return optimizer
    
    def _setup_scheduler(self) -> torch.optim.lr_scheduler._LRScheduler:
        """Setup learning rate scheduler"""
        if getattr(self.train_config, 'lr_scheduler', 'linear') == "cosine":
            # Warmup + Cosine annealing
            warmup_scheduler = LinearLR(
                self.optimizer,
                start_factor=0.1,
                end_factor=1.0,
                total_iters=getattr(self.train_config, 'warmup_steps', 1000)
            )
            
            cosine_scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=getattr(self.train_config, 'lr_decay_iters', 10000) - getattr(self.train_config, 'warmup_steps', 1000),
                eta_min=getattr(self.train_config, 'min_lr', 1e-6)
            )
            
            scheduler = SequentialLR(
                self.optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[getattr(self.train_config, 'warmup_steps', 1000)]
            )
        else:
            # Simple linear warmup
            scheduler = LinearLR(
                self.optimizer,
                start_factor=0.1,
                end_factor=1.0,
                total_iters=getattr(self.train_config, 'warmup_steps', 1000)
            )
        
        return scheduler
    
    def _setup_wandb(self):
        """Setup Weights & Biases logging"""
        config = {
            **asdict(self.model_config),
            **asdict(self.train_config),
            **asdict(self.data_config)
        }
        
        wandb.init(
            project=self.train_config.wandb_project,
            name=self.train_config.wandb_run_name,
            config=config,
            resume="allow"
        )
        
        # Watch model
        wandb.watch(self.raw_model, log="all", log_freq=1000)
    
    def _setup_datasets(self):
        """Setup training and validation datasets"""
        data_dir = 'data'  # Assume processed data is in data/ directory
        
        if os.path.exists(os.path.join(data_dir, 'train.bin')):
            self.train_dataset = ChildrenStoriesDataset(
                data_dir, 
                self.model_config.block_size, 
                'train'
            )
        
        if os.path.exists(os.path.join(data_dir, 'val.bin')):
            self.val_dataset = ChildrenStoriesDataset(
                data_dir, 
                self.model_config.block_size, 
                'val'
            )
    
    def get_batch(self, split: str = 'train') -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a batch of data"""
        dataset = self.train_dataset if split == 'train' else self.val_dataset
        
        if dataset is None:
            raise ValueError(f"Dataset for split '{split}' not found")
        
        # Sample random indices
        indices = torch.randint(0, len(dataset), (getattr(self.train_config, 'batch_size', 16),))
        
        # Get batch
        batch_x = []
        batch_y = []
        
        for idx in indices:
            x, y = dataset[idx]
            batch_x.append(x)
            batch_y.append(y)
        
        # Stack and move to device
        x = torch.stack(batch_x).to(self.device)
        y = torch.stack(batch_y).to(self.device)
        
        return x, y
    
    def train_step(self) -> Dict[str, float]:
        """Single training step"""
        self.model.train()
        
        # Get batch
        x, y = self.get_batch('train')
        
        # Forward pass with mixed precision (disabled for MPS)
        if self.scaler is not None and self.device.type == 'cuda':
            with torch.cuda.amp.autocast():
                logits, loss = self.model(x, y)
        else:
            logits, loss = self.model(x, y)
        
        # Normalize loss for gradient accumulation
        loss = loss / getattr(self.train_config, 'gradient_accumulation_steps', 4)
        
        # Backward pass
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        return {'loss': loss.item() * getattr(self.train_config, 'gradient_accumulation_steps', 4)}
    
    def optimizer_step(self) -> Dict[str, float]:
        """Optimizer step with gradient clipping"""
        # Gradient clipping
        if self.scaler is not None:
            self.scaler.unscale_(self.optimizer)
        
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), 
            getattr(self.train_config, 'max_grad_norm', 1.0)
        )
        
        # Optimizer step
        if self.scaler is not None:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        
        self.optimizer.zero_grad()
        self.scheduler.step()
        
        return {'grad_norm': grad_norm.item()}
    
    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """Evaluate model on validation set"""
        if self.val_dataset is None:
            return {'val_loss': 0.0}
        
        self.model.eval()
        
        total_loss = 0.0
        num_batches = getattr(self.train_config, 'eval_iters', 10)
        
        for _ in range(num_batches):
            x, y = self.get_batch('val')
            
            if self.scaler is not None:
                with torch.cuda.amp.autocast():
                    logits, loss = self.model(x, y)
            else:
                logits, loss = self.model(x, y)
            
            total_loss += loss.item()
        
        avg_loss = total_loss / num_batches
        return {'val_loss': avg_loss}
    
    def save_checkpoint(self, is_best: bool = False, extra_info: Optional[Dict] = None):
        """Save model checkpoint"""
        if self.is_distributed and dist.get_rank() != 0:
            return  # Only save on rank 0
        
        checkpoint = {
            'model_state_dict': self.raw_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'current_iter': self.current_iter,
            'best_val_loss': self.best_val_loss,
            'model_config': asdict(self.model_config),
            'train_config': asdict(self.train_config),
            'metrics': self.metrics
        }
        
        if extra_info:
            checkpoint.update(extra_info)
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(self.checkpoint_dir, f'checkpoint_iter_{self.current_iter}.pt')
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best_model.pt')
            torch.save(checkpoint, best_path)
            print(f"Saved best model with val_loss: {self.best_val_loss:.4f}")
        
        # Cleanup old checkpoints
        self._cleanup_checkpoints()
    
    def _cleanup_checkpoints(self):
        """Remove old checkpoints to save disk space"""
        checkpoint_files = [f for f in os.listdir(self.checkpoint_dir) if f.startswith('checkpoint_iter_')]
        checkpoint_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
        
        # Keep only the most recent checkpoints
        keep_checkpoints = getattr(self.train_config, 'keep_checkpoints', 5)
        if len(checkpoint_files) > keep_checkpoints:
            for file_to_remove in checkpoint_files[:-keep_checkpoints]:
                os.remove(os.path.join(self.checkpoint_dir, file_to_remove))
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        print(f"Loading checkpoint from {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model state
        self.raw_model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Load training state
        self.current_iter = checkpoint['current_iter']
        self.best_val_loss = checkpoint['best_val_loss']
        self.metrics = checkpoint.get('metrics', self.metrics)
        
        print(f"Resumed from iteration {self.current_iter}")
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get memory usage statistics"""
        stats = {}
        
        if torch.cuda.is_available():
            stats['gpu_memory_allocated'] = torch.cuda.memory_allocated() / 1e9
            stats['gpu_memory_cached'] = torch.cuda.memory_reserved() / 1e9
        
        # CPU memory
        process = psutil.Process()
        stats['cpu_memory_mb'] = process.memory_info().rss / 1e6
        
        return stats
    
    def log_metrics(self, metrics: Dict[str, float], step: int):
        """Log metrics to console and wandb"""
        # Console logging
        if step % getattr(self.train_config, 'log_interval', 100) == 0:
            memory_stats = self.get_memory_usage()
            
            log_str = f"Step {step:6d} | "
            log_str += f"Loss: {metrics.get('loss', 0):.4f} | "
            log_str += f"LR: {self.scheduler.get_last_lr()[0]:.2e} | "
            if 'val_loss' in metrics:
                log_str += f"Val: {metrics['val_loss']:.4f} | "
            if 'gpu_memory_allocated' in memory_stats:
                log_str += f"GPU: {memory_stats['gpu_memory_allocated']:.1f}GB"
            
            print(log_str)
        
        # Wandb logging
        if self.use_wandb:
            log_dict = {
                'step': step,
                'learning_rate': self.scheduler.get_last_lr()[0],
                **metrics,
                **self.get_memory_usage()
            }
            wandb.log(log_dict, step=step)
    
    def train(self):
        """Main training loop"""
        print("Starting training...")
        max_iters = getattr(self.train_config, 'max_iters', 5000)
        batch_size = getattr(self.train_config, 'batch_size', 16)
        grad_acc_steps = getattr(self.train_config, 'gradient_accumulation_steps', 4)
        print(f"Max iterations: {max_iters}")
        print(f"Batch size: {batch_size}")
        print(f"Gradient accumulation steps: {grad_acc_steps}")
        print(f"Effective batch size: {batch_size * grad_acc_steps}")
        
        self.model.train()
        start_time = time.time()
        
        # Training loop
        pbar = tqdm(range(self.current_iter, max_iters), desc="Training")
        
        for iter_num in pbar:
            self.current_iter = iter_num
            
            # Accumulate gradients
            total_loss = 0.0
            for micro_step in range(grad_acc_steps):
                step_metrics = self.train_step()
                total_loss += step_metrics['loss']
            
            # Optimizer step
            opt_metrics = self.optimizer_step()
            
            # Store metrics
            avg_loss = total_loss / grad_acc_steps
            self.metrics['train_loss'].append(avg_loss)
            self.metrics['learning_rates'].append(self.scheduler.get_last_lr()[0])
            self.metrics['grad_norm'].append(opt_metrics['grad_norm'])
            
            # Evaluation
            eval_metrics = {}
            eval_interval = getattr(self.train_config, 'eval_interval', 500)
            if iter_num % eval_interval == 0 and iter_num > 0:
                eval_metrics = self.evaluate()
                self.metrics['val_loss'].append(eval_metrics['val_loss'])
                
                # Save best model
                if eval_metrics['val_loss'] < self.best_val_loss:
                    self.best_val_loss = eval_metrics['val_loss']
                    self.save_checkpoint(is_best=True)
            
            # Logging
            log_metrics = {'loss': avg_loss, **opt_metrics, **eval_metrics}
            self.log_metrics(log_metrics, iter_num)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{avg_loss:.4f}",
                'lr': f"{self.scheduler.get_last_lr()[0]:.2e}",
                'val': f"{eval_metrics.get('val_loss', 0):.4f}" if eval_metrics else "N/A"
            })
            
            # Checkpointing
            save_interval = getattr(self.train_config, 'save_checkpoint_every', 1000)
            if iter_num % save_interval == 0 and iter_num > 0:
                self.save_checkpoint()
            
            # Cleanup GPU memory periodically
            if iter_num % 1000 == 0:
                torch.cuda.empty_cache()
                gc.collect()
        
        # Final checkpoint
        self.save_checkpoint()
        
        # Training summary
        total_time = time.time() - start_time
        print(f"\nTraining completed!")
        print(f"Total time: {total_time:.2f} seconds")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        
        # Close wandb
        if self.use_wandb:
            wandb.finish()
    
    def plot_metrics(self, save_path: Optional[str] = None):
        """Plot training metrics"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Training loss
        if self.metrics['train_loss']:
            axes[0, 0].plot(self.metrics['train_loss'])
            axes[0, 0].set_title('Training Loss')
            axes[0, 0].set_xlabel('Iteration')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].grid(True)
        
        # Validation loss
        if self.metrics['val_loss']:
            eval_steps = np.arange(0, len(self.metrics['val_loss'])) * getattr(self.train_config, 'eval_interval', 500)
            axes[0, 1].plot(eval_steps, self.metrics['val_loss'])
            axes[0, 1].set_title('Validation Loss')
            axes[0, 1].set_xlabel('Iteration')
            axes[0, 1].set_ylabel('Loss')
            axes[0, 1].grid(True)
        
        # Learning rate
        if self.metrics['learning_rates']:
            axes[1, 0].plot(self.metrics['learning_rates'])
            axes[1, 0].set_title('Learning Rate')
            axes[1, 0].set_xlabel('Iteration')
            axes[1, 0].set_ylabel('LR')
            axes[1, 0].set_yscale('log')
            axes[1, 0].grid(True)
        
        # Gradient norm
        if self.metrics['grad_norm']:
            axes[1, 1].plot(self.metrics['grad_norm'])
            axes[1, 1].set_title('Gradient Norm')
            axes[1, 1].set_xlabel('Iteration')
            axes[1, 1].set_ylabel('Grad Norm')
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(os.path.join(self.checkpoint_dir, 'training_metrics.png'), 
                       dpi=300, bbox_inches='tight')
        
        plt.show()


def create_trainer_from_configs(
    model_config: ModelConfig,
    training_config: TrainingConfig, 
    data_config: DataConfig,
    checkpoint_dir: str = 'checkpoints'
) -> AdvancedTrainer:
    """Create trainer from configuration objects"""
    
    # Convert to GPTOSSAdvancedConfig
    gpt_config = GPTOSSAdvancedConfig(
        vocab_size=model_config.vocab_size,
        n_layer=model_config.n_layer,
        n_head=model_config.n_head,
        n_embd=model_config.n_embd,
        block_size=model_config.block_size,
        # dropout removed in GPT-OSS
        bias=model_config.bias,
        num_key_value_heads=model_config.num_key_value_heads,
        sliding_window=model_config.sliding_window,
        use_attention_sinks=model_config.use_attention_sinks,
        head_dim=model_config.head_dim,
        rope_theta=model_config.rope_theta,
        rope_scaling_factor=model_config.rope_scaling_factor,
        rope_ntk_alpha=model_config.rope_ntk_alpha,
        rope_ntk_beta=model_config.rope_ntk_beta,
        num_experts=model_config.num_experts,
        experts_per_token=model_config.experts_per_token,
        expert_capacity=model_config.expert_capacity,
        moe_aux_loss_coeff=model_config.moe_aux_loss_coeff,
        moe_layers=model_config.moe_layers,
        multi_token_predict=model_config.multi_token_predict,
        use_swiglu=model_config.use_swiglu,
        use_rmsnorm=model_config.use_rmsnorm,
        swiglu_limit=model_config.swiglu_limit,
        use_quantization=model_config.use_quantization,
        quantization_bits=model_config.quantization_bits
    )
    
    # Create model
    model = GPTOSSAdvanced(gpt_config)
    
    # Create trainer
    trainer = AdvancedTrainer(
        model=model,
        train_config=training_config,
        model_config=model_config,
        data_config=data_config,
        checkpoint_dir=checkpoint_dir
    )
    
    return trainer
