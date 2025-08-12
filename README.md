# GPT-OSS Advanced Children's Stories Model

A state-of-the-art language model based on OpenAI's GPT-OSS architecture, specifically optimized for generating engaging and educational children's stories. This project combines the advanced features of GPT-OSS with the structured approach of educational models like Tiny Stories and DeepSeek.

## 🌟 Features

### GPT-OSS Architecture
- **Mixture of Experts (MoE)**: Efficient expert routing with configurable number of experts
- **Extended Context Length**: Support for up to 8,192 tokens (scalable to GPT-OSS's 128k)
- **Sliding Window Attention**: Alternating dense and sparse attention patterns
- **Grouped Query Attention**: Efficient attention with shared key-value heads
- **Advanced Optimizations**: SwiGLU activation, RMSNorm, attention sinks

### Educational Focus
- **Children's Story Generation**: Optimized for age-appropriate, engaging content
- **Multi-token Prediction**: Improved training efficiency with simultaneous token prediction
- **Clean Architecture**: Educational code structure following Tiny Stories format
- **Comprehensive Documentation**: Easy to understand and modify

### Production Ready
- **Mixed Precision Training**: Automatic mixed precision with gradient scaling
- **Distributed Training**: Multi-GPU support with DistributedDataParallel
- **Advanced Sampling**: Top-k, top-p, temperature, and repetition penalty
- **Monitoring**: Weights & Biases integration and comprehensive metrics

## 🚀 Quick Start

### Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd GPT-OSS-Children-Stories-Advanced
```

2. Run the setup script:
```bash
chmod +x setup.sh
./setup.sh
```

The setup script will:
- Create a virtual environment
- Install all dependencies including GPT-OSS specific packages
- Set up the directory structure
- Verify the installation
- Optionally download sample data

### Training a Model

#### Quick Training (Small Model)
```bash
# Train a small model for testing
python3 src/run_training.py \
    --model-size small \
    --training-preset fast \
    --max-iters 5000 \
    --process-data \
    --use-wandb
```

#### Production Training
```bash
# Train a larger model for production
python3 src/run_training.py \
    --model-size medium \
    --training-preset standard \
    --max-iters 50000 \
    --batch-size 16 \
    --gradient-accumulation-steps 4 \
    --process-data \
    --use-wandb \
    --compile-model \
    --mixed-precision
```

### Generating Stories

#### Interactive Mode
```bash
python3 src/generate.py \
    --model-path checkpoints/best_model.pt \
    --interactive
```

#### Batch Generation
```bash
python3 src/generate.py \
    --model-path checkpoints/best_model.pt \
    --prompt "Once upon a time in a magical forest" \
    --max-tokens 200 \
    --temperature 0.8 \
    --num-samples 5
```

## 📊 Model Configurations

### Available Model Sizes

| Size | Layers | Heads | Embedding | Context | Experts | Parameters |
|------|--------|-------|-----------|---------|---------|------------|
| tiny | 6 | 8 | 512 | 2048 | 4 | ~15M |
| small | 12 | 12 | 768 | 4096 | 8 | ~50M |
| medium | 24 | 16 | 1024 | 8192 | 16 | ~200M |
| large | 36 | 20 | 1280 | 8192 | 32 | ~500M |

### Training Presets

| Preset | Max Iters | Batch Size | Use Case |
|--------|-----------|------------|----------|
| debug | 1000 | 4 | Quick testing |
| fast | 10000 | 8 | Rapid prototyping |
| standard | 50000 | 16 | Normal training |
| production | 100000 | 32 | High-quality models |

## 🏗️ Architecture Details

### GPT-OSS Innovations

1. **Mixture of Experts (MoE)**
   - Configurable number of experts per layer
   - Top-k routing with load balancing
   - Auxiliary loss for expert utilization

2. **Advanced Attention Mechanisms**
   - Sliding window attention for efficiency
   - Attention sinks for improved long-context handling
   - Grouped query attention with shared KV heads

3. **Modern Optimizations**
   - SwiGLU activation function
   - RMSNorm instead of LayerNorm
   - Rotary positional embeddings (RoPE) with YaRN scaling

4. **Multi-token Prediction**
   - Simultaneous prediction of multiple future tokens
   - Improved training efficiency and convergence

### Key Differences from GPT-2

| Feature | GPT-2 | GPT-OSS Advanced |
|---------|-------|------------------|
| Vocabulary | 50,257 | 201,088 |
| Context Length | 1024-2048 | 8192+ |
| Attention | Standard MHA | Sliding Window + Sinks |
| MLP | Dense | Mixture of Experts |
| Activation | GELU | SwiGLU |
| Normalization | LayerNorm | RMSNorm |
| Position Encoding | Learned | RoPE with YaRN |

## 📁 Project Structure

```
GPT-OSS-Children-Stories-Advanced/
├── src/
│   ├── model/
│   │   └── gpt_oss_advanced.py      # Main model architecture
│   ├── training/
│   │   └── advanced_trainer.py      # Training system
│   ├── data/
│   │   └── data_processor.py        # Data processing pipeline
│   ├── config.py                    # Configuration classes
│   ├── generate.py                  # Text generation script
│   └── run_training.py              # Main training script
├── scripts/
│   ├── train_small_model.sh         # Quick training script
│   ├── generate_stories.sh          # Story generation script
│   └── process_data.sh              # Data processing script
├── checkpoints/                     # Model checkpoints
├── data/                           # Processed training data
├── logs/                           # Training logs
├── outputs/                        # Generated stories
├── requirements.txt                # Python dependencies
├── setup.sh                       # Setup script
└── README.md                      # This file
```

## 🔧 Configuration

### Model Configuration
```python
from src.config import ModelConfig

config = ModelConfig(
    vocab_size=201088,        # GPT-OSS vocabulary
    n_layer=12,              # Number of layers
    n_head=16,               # Attention heads
    n_embd=768,              # Embedding dimension
    block_size=8192,         # Context length
    num_experts=8,           # MoE experts
    experts_per_token=2,     # Active experts per token
    sliding_window=256,      # Sliding window size
    multi_token_predict=2,   # Multi-token prediction
    use_swiglu=True,         # SwiGLU activation
    use_rmsnorm=True,        # RMSNorm
    use_attention_sinks=True # Attention sinks
)
```

### Training Configuration
```python
from src.config import TrainingConfig

config = TrainingConfig(
    max_iters=50000,
    batch_size=16,
    gradient_accumulation_steps=4,
    learning_rate=3e-4,
    use_mixed_precision=True,
    compile_model=True,
    use_wandb=True
)
```

## 📊 Training Monitoring

The model supports comprehensive monitoring through:

- **Weights & Biases**: Real-time loss curves, learning rates, and system metrics
- **TensorBoard**: Alternative logging backend
- **Console Logging**: Detailed progress information
- **Automatic Plotting**: Training curves saved as images

### Key Metrics Tracked
- Training and validation loss
- Learning rate schedule
- Gradient norms
- Memory usage (GPU/CPU)
- MoE auxiliary loss
- Multi-token prediction accuracy
- Tokens per second throughput

## 🎯 Use Cases

### Educational Applications
- **Interactive Storytelling**: Generate personalized stories for children
- **Reading Comprehension**: Create stories with embedded questions
- **Creative Writing**: Assist children in story creation
- **Language Learning**: Generate stories at appropriate difficulty levels

### Research Applications
- **MoE Architecture Studies**: Experiment with different expert configurations
- **Long Context Modeling**: Test extended context capabilities
- **Multi-token Prediction**: Research parallel decoding strategies
- **Efficient Training**: Study advanced optimization techniques

## 🔬 Advanced Features

### Custom Tokenization
The model uses GPT-OSS's harmony tokenizer with 201,088 vocabulary size:
```python
from openai_harmony import load_harmony_encoding, HarmonyEncodingName
tokenizer = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
```

### Distributed Training
Multi-GPU training with automatic gradient synchronization:
```bash
torchrun --nproc-per-node=4 src/run_training.py \
    --use-ddp \
    --model-size large \
    --training-preset production
```

### Model Compilation
PyTorch 2.0+ compilation for improved performance:
```bash
python3 src/run_training.py \
    --compile-model \
    --model-size medium
```

## 🚨 Hardware Requirements

### Minimum Requirements
- **GPU**: 8GB VRAM (for tiny/small models)
- **RAM**: 16GB system memory
- **Storage**: 50GB free space

### Recommended Requirements
- **GPU**: 24GB VRAM (RTX 4090, A100)
- **RAM**: 32GB+ system memory
- **Storage**: 500GB+ SSD

### Production Requirements
- **GPU**: 80GB VRAM (H100, A100 80GB)
- **RAM**: 128GB+ system memory
- **Storage**: 1TB+ NVMe SSD

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines for:
- Code style and formatting
- Testing requirements
- Documentation standards
- Pull request process

## 📄 License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **OpenAI**: For the GPT-OSS architecture and research
- **DeepSeek**: For advanced optimization techniques
- **Tiny Stories**: For educational model design inspiration
- **Hugging Face**: For datasets and model infrastructure
- **PyTorch Team**: For the excellent deep learning framework

## 📚 References

1. [Introducing GPT-OSS](https://openai.com/index/introducing-gpt-oss/) - OpenAI Blog
2. [GPT-OSS Model Card](https://openai.com/index/gpt-oss-model-card/) - Technical Specifications
3. [GPT-OSS Cookbook](https://cookbook.openai.com/topic/gpt-oss) - Usage Guides
4. [Harmony Response Format](https://github.com/openai/harmony) - Tokenization Details

---

**Built with ❤️ for educational AI and children's storytelling**
