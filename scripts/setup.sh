#!/bin/bash
# ============================================================
# Potato-ACHIS: Quick Start Script
# ============================================================
# Usage: ./scripts/setup.sh [--cuda|--cpu]
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}🥔 Potato-ACHIS Setup${NC}"
echo "================================"

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo -e "${YELLOW}Installing uv...${NC}"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
fi

echo -e "${GREEN}✓ uv found: $(uv --version)${NC}"

# Parse arguments
CUDA_FLAG=""
if [[ "$1" == "--cpu" ]]; then
    echo -e "${YELLOW}Installing CPU-only PyTorch${NC}"
    CUDA_FLAG="--index https://download.pytorch.org/whl/cpu"
elif [[ "$1" == "--cuda" ]] || [[ -z "$1" ]]; then
    echo -e "${GREEN}Installing CUDA-enabled PyTorch${NC}"
fi

cd "$PROJECT_DIR"

# Sync dependencies
echo -e "\n${GREEN}📦 Installing dependencies...${NC}"
uv sync --all-extras

# Setup pre-commit hooks
echo -e "\n${GREEN}🔧 Setting up pre-commit hooks...${NC}"
uv run pre-commit install

# Verify installation
echo -e "\n${GREEN}✅ Verifying installation...${NC}"
uv run python -c "
import torch
import torchvision
import timm

print(f'  PyTorch: {torch.__version__}')
print(f'  TorchVision: {torchvision.__version__}')
print(f'  TIMM: {timm.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  CUDA device: {torch.cuda.get_device_name(0)}')
"

echo -e "\n${GREEN}🎉 Setup complete!${NC}"
echo ""
echo "Quick commands:"
echo "  uv run python src/train.py --help    # Training help"
echo "  uv run python src/eval.py --help     # Evaluation help"
echo "  uv run pytest                        # Run tests"
echo "  uv run ruff check src/               # Lint code"
