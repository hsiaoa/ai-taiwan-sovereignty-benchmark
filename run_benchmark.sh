#!/bin/bash
# Taiwan Sovereignty Benchmark - Quick Run Script
# 台灣主權基準測試 - 快速執行腳本

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}🇹🇼 Taiwan Sovereignty Benchmark${NC}"
echo "=================================="

# Check Python
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: Python 3 is required${NC}"
    exit 1
fi

# Check AWS credentials
if ! aws sts get-caller-identity &> /dev/null; then
    echo -e "${YELLOW}Warning: AWS credentials not configured${NC}"
    echo "Please run: aws configure"
    echo "Or set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY"
fi

# Install dependencies if needed
if ! python3 -c "import boto3" 2>/dev/null; then
    echo "Installing dependencies..."
    pip3 install -r requirements.txt
fi

# Run benchmark
echo ""
echo "Starting benchmark..."
python3 src/bedrock_benchmark.py "$@"
