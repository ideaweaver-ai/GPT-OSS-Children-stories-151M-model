#!/bin/bash

# GPT-OSS Advanced Children's Stories Model Setup Script
# Combines setup approaches from GPT-OSS, DeepSeek, and Tiny Stories

set -e  # Exit on any error

echo "=========================================="
echo "GPT-OSS Advanced Children's Stories Setup"
echo "=========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running in virtual environment
check_virtual_env() {
    if [[ "$VIRTUAL_ENV" == "" ]]; then
        print_warning "Not running in a virtual environment!"
        read -p "Do you want to create and activate a virtual environment? (y/n): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            print_status "Creating virtual environment..."
            python3 -m venv gpt-oss-env
            source gpt-oss-env/bin/activate
            print_success "Virtual environment created and activated"
        else
            print_warning "Continuing without virtual environment (not recommended)"
        fi
    else
        print_success "Virtual environment detected: $VIRTUAL_ENV"
    fi
}

# Check Python version
check_python_version() {
    print_status "Checking Python version..."
    
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 is not installed!"
        exit 1
    fi
    
    python_version=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    required_version="3.8"
    
    if python3 -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)"; then
        print_success "Python version $python_version is compatible"
    else
        print_error "Python version $python_version is too old. Requires Python >= $required_version"
        exit 1
    fi
}

# Check CUDA availability
check_cuda() {
    print_status "Checking CUDA availability..."
    
    if command -v nvidia-smi &> /dev/null; then
        gpu_info=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits | head -1)
        print_success "CUDA GPU detected: $gpu_info"
        export CUDA_AVAILABLE=1
    else
        print_warning "CUDA not detected. Will use CPU training (much slower)"
        export CUDA_AVAILABLE=0
    fi
}

# Install PyTorch with appropriate CUDA version
install_pytorch() {
    print_status "Installing PyTorch..."
    
    if [[ $CUDA_AVAILABLE -eq 1 ]]; then
        # Check CUDA version
        cuda_version=$(nvidia-smi | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+' | head -1)
        print_status "Detected CUDA version: $cuda_version"
        
        if [[ $(echo "$cuda_version >= 11.8" | bc -l) -eq 1 ]]; then
            pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
        elif [[ $(echo "$cuda_version >= 11.7" | bc -l) -eq 1 ]]; then
            pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
        else
            print_warning "CUDA version $cuda_version might not be fully supported"
            pip install torch torchvision torchaudio
        fi
    else
        # CPU-only PyTorch
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    fi
    
    print_success "PyTorch installed successfully"
}

# Install dependencies
install_dependencies() {
    print_status "Installing Python dependencies..."
    
    # Upgrade pip first
    python3 -m pip install --upgrade pip
    
    # Install core dependencies
    pip install -r requirements.txt
    
    # Install optional dependencies based on system
    if [[ $CUDA_AVAILABLE -eq 1 ]]; then
        print_status "Installing CUDA-specific optimizations..."
        
        # Try to install flash-attention (optional)
        print_status "Attempting to install flash-attention..."
        pip install flash-attn --no-build-isolation || print_warning "Flash attention installation failed (optional)"
        
        # Try to install triton (optional)
        print_status "Attempting to install triton..."
        pip install triton || print_warning "Triton installation failed (optional)"
    fi
    
    print_success "Dependencies installed successfully"
}

# Create directory structure
create_directories() {
    print_status "Creating directory structure..."
    
    directories=(
        "data"
        "checkpoints"
        "logs"
        "outputs"
        "configs"
        "scripts"
    )
    
    for dir in "${directories[@]}"; do
        mkdir -p "$dir"
        print_status "Created directory: $dir"
    done
    
    print_success "Directory structure created"
}

# Download sample data (optional)
download_sample_data() {
    read -p "Do you want to download and process sample data? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_status "Processing sample dataset..."
        
        python3 -c "
from src.data.data_processor import AdvancedDataProcessor
from src.config import DataConfig

config = DataConfig(
    dataset_name='ajibawa-2023/Children-Stories-Collection',
    max_seq_length=512,  # Smaller for demo
    data_split_ratios=[0.8, 0.1, 0.1]
)

processor = AdvancedDataProcessor(config)
try:
    results = processor.process_dataset('data')
    print('Sample data processed successfully!')
    print(f'Results: {results}')
except Exception as e:
    print(f'Error processing data: {e}')
    print('You can process data later using: python3 src/run_training.py --process-data')
"
        print_success "Sample data processing completed"
    else
        print_status "Skipping sample data download"
    fi
}

# Create example scripts
create_example_scripts() {
    print_status "Creating example scripts..."
    
    # Training script
    cat > scripts/train_small_model.sh << 'EOF'
#!/bin/bash
# Train a small model for testing
python3 src/run_training.py \
    --model-size small \
    --training-preset fast \
    --max-iters 5000 \
    --batch-size 8 \
    --eval-interval 500 \
    --process-data \
    --use-wandb \
    --generate-samples
EOF

    # Generation script
    cat > scripts/generate_stories.sh << 'EOF'
#!/bin/bash
# Generate stories using trained model
python3 src/generate.py \
    --model-path checkpoints/best_model.pt \
    --prompt "Once upon a time" \
    --max-tokens 200 \
    --temperature 0.8 \
    --num-samples 3 \
    --interactive
EOF

    # Data processing script
    cat > scripts/process_data.sh << 'EOF'
#!/bin/bash
# Process data only
python3 src/run_training.py \
    --process-data \
    --max-seq-length 1024 \
    --dataset-name "ajibawa-2023/Children-Stories-Collection"
EOF

    chmod +x scripts/*.sh
    print_success "Example scripts created in scripts/ directory"
}

# Verify installation
verify_installation() {
    print_status "Verifying installation..."
    
    # Test Python imports
    python3 -c "
import torch
import transformers
import datasets
import tiktoken
import numpy as np
import matplotlib

print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA device: {torch.cuda.get_device_name()}')
print(f'Transformers version: {transformers.__version__}')
print(f'Datasets version: {datasets.__version__}')

# Test model import
try:
    from src.model.gpt_oss_advanced import GPTOSSAdvanced, GPTOSSAdvancedConfig
    from src.config import ModelConfig, TrainingConfig
    print('✓ Model imports successful')
except Exception as e:
    print(f'✗ Model import failed: {e}')
    exit(1)

# Test config creation
try:
    config = GPTOSSAdvancedConfig()
    model = GPTOSSAdvanced(config)
    print(f'✓ Model creation successful ({model.get_num_params():,} parameters)')
except Exception as e:
    print(f'✗ Model creation failed: {e}')
    exit(1)

print('\\n✓ Installation verification successful!')
"
    
    if [[ $? -eq 0 ]]; then
        print_success "Installation verified successfully!"
    else
        print_error "Installation verification failed!"
        exit 1
    fi
}

# Print usage information
print_usage() {
    print_success "Setup completed successfully!"
    echo
    echo "=========================================="
    echo "Usage Information"
    echo "=========================================="
    echo
    echo "1. Train a model:"
    echo "   ./scripts/train_small_model.sh"
    echo "   or"
    echo "   python3 src/run_training.py --model-size small --training-preset fast"
    echo
    echo "2. Generate stories:"
    echo "   ./scripts/generate_stories.sh"
    echo "   or"
    echo "   python3 src/generate.py --model-path checkpoints/best_model.pt --interactive"
    echo
    echo "3. Process data only:"
    echo "   ./scripts/process_data.sh"
    echo "   or"
    echo "   python3 src/run_training.py --process-data"
    echo
    echo "4. Available model sizes: tiny, small, medium, large"
    echo "5. Available training presets: debug, fast, standard, production"
    echo
    echo "For more options, run:"
    echo "   python3 src/run_training.py --help"
    echo "   python3 src/generate.py --help"
    echo
    echo "Documentation and examples are in the README.md file."
    echo "=========================================="
}

# Main setup function
main() {
    print_status "Starting setup process..."
    
    check_virtual_env
    check_python_version
    check_cuda
    install_pytorch
    install_dependencies
    create_directories
    create_example_scripts
    verify_installation
    
    # Optional data download
    download_sample_data
    
    print_usage
}

# Handle script interruption
trap 'print_error "Setup interrupted by user"; exit 1' INT

# Run main setup
main

print_success "Setup completed successfully! 🎉"
print_status "You can now start training your GPT-OSS Advanced model for children's stories!"
