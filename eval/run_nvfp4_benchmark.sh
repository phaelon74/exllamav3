#!/bin/bash

# Script to run NVFP4 perplexity benchmark with proper memory management
# This handles the memory fragmentation issues with compressed-tensors decompression

echo "========================================="
echo "NVFP4 Perplexity Benchmark Runner"
echo "========================================="
echo ""

# Set PyTorch memory allocator to reduce fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Clear CUDA cache
echo "Clearing CUDA cache..."
python -c "import torch; torch.cuda.empty_cache()"

echo ""
echo "Running benchmark with reduced context length (1024 tokens)..."
echo "This avoids OOM errors during NVFP4 decompression."
echo ""

# Run the benchmark
python eval/compare_q.py \
    -d eval/spec/wiki2_behemoth_short.json \
    -m eval/spec/behemoth_nvfp4_transformers.json

echo ""
echo "========================================="
echo "Benchmark Complete!"
echo "========================================="


