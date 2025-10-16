# vLLM Integration for ExLlamaV3 Benchmarking

This guide explains how to use the vLLM integration to benchmark NVFP4 Compressed Tensors and other vLLM-compatible models using the ExLlamaV3 comparison framework.

## Overview

The vLLM integration (`compare_q_vllm.py`) allows you to:
- Benchmark NVFP4 quantized models (Compressed Tensors format)
- Compare vLLM models against EXL2, EXL3, GGUF, and other quantization formats
- Calculate perplexity on standard datasets (WikiText-2)
- Generate comparison plots

## Requirements

### Installation

```bash
# Install vLLM
pip install vllm

# Install comparison framework dependencies
pip install -r requirements_eval.txt
```

### GPU Requirements
- NVIDIA GPU with adequate VRAM for your model
- CUDA-compatible GPU (vLLM requirement)
- For NVFP4: NVIDIA Blackwell architecture recommended (though other GPUs may work)

## Usage

### 1. Prepare Your Model Specification

Create a JSON file specifying your vLLM model(s). See `eval/spec/example_vllm_nvfp4.json`:

```json
[
    {
        "load_fn": "vllm",
        "fwd_fn": "vllm",
        "label": "vLLM NVFP4",
        "model_dir": "/path/to/your/nvfp4/compressed_tensor/model"
    }
]
```

**Fields:**
- `load_fn`: Must be `"vllm"`
- `fwd_fn`: Must be `"vllm"`
- `label`: Display name for plots (use format like "vLLM NVFP4" or "VLLM FP4")
- `model_dir`: Path to model directory or HuggingFace model ID

### 2. Prepare Data Specification

Use an existing data spec or create your own:

```json
{
    "tokenize_fn": "transformers",
    "tokenizer_dir": "/path/to/tokenizer/or/base/model",
    "dataset": "wiki2",
    "eval_stride": 512,
    "eval_len": 2048,
    "max_rows": 20
}
```

**Note:** The tokenizer should match your model. For Llama models, use the base Llama tokenizer path.

### 3. Run the Benchmark

```bash
cd eval

# Basic perplexity test
python compare_q.py \
    -d spec/wiki2_llama3.json \
    -m spec/example_vllm_nvfp4.json

# With plotting
python compare_q.py \
    -d spec/wiki2_llama3.json \
    -m spec/example_vllm_nvfp4.json \
    -p \
    -t "Llama-3.1-8B NVFP4 Comparison"

# Compare vLLM with other formats
python compare_q.py \
    -d spec/wiki2_llama3.json \
    -m spec/example_vllm_nvfp4.json spec/llama3.1-8b-instruct_exl3.json \
    -p \
    -t "NVFP4 vs EXL3"
```

### 4. Advanced Options

```bash
# Clear cache before running
python compare_q.py \
    -d spec/wiki2_llama3.json \
    -m spec/example_vllm_nvfp4.json \
    -cc

# Use VRAM as X-axis instead of BPW
python compare_q.py \
    -d spec/wiki2_llama3.json \
    -m spec/example_vllm_nvfp4.json \
    -p \
    -v

# Filter specific models
python compare_q.py \
    -d spec/wiki2_llama3.json \
    -m spec/*.json \
    -mask "vLLM;EXL3" \
    -p

# Dark mode plotting
python compare_q.py \
    -d spec/wiki2_llama3.json \
    -m spec/example_vllm_nvfp4.json \
    -p \
    -dark
```

## Understanding the Output

### Console Output
```
Loading vLLM model from: /path/to/model
Loaded vLLM model: 4.00 bpw (decoder), 4.00 bpw (head)
Loading dataset: wiki2
Testing: /path/to/model (vLLM NVFP4)
Evaluating: 100% |████████████████████| 20/20
Perplexity: 8.523456
```

### JSON Results
```json
{
    "label": "vLLM NVFP4",
    "layer_bpw": 4.0,
    "head_bpw": 4.0,
    "vram_gb": 3.45,
    "ppl": 8.523456
}
```

### Plot
The plot will show:
- **X-axis**: Bits per weight (decoder only) or VRAM usage
- **Y-axis**: Perplexity (lower is better)
- **Color coding**: Different quantization methods get different colors
- **Lines**: Connect models from the same quantization family

## Important Notes

### Limitations

1. **Batch Size**: Currently only supports batch_size=1 due to vLLM API constraints
2. **BPW Estimation**: Bits-per-weight calculation is approximate for vLLM models since vLLM doesn't expose detailed layer information
3. **Logits vs Logprobs**: vLLM returns log probabilities rather than raw logits. The implementation converts these appropriately for perplexity calculation
4. **Memory**: vLLM models require significant VRAM. Ensure `gpu_memory_utilization` is set appropriately (default: 0.9)

### Troubleshooting

**Error: "CUDA out of memory"**
- Reduce `max_rows` in your data spec
- Adjust GPU memory allocation in `compare_q_vllm.py` (line 106: `gpu_memory_utilization`)

**Error: "Model not found"**
- Verify `model_dir` path is correct
- Ensure vLLM can load your model format
- For HuggingFace models, ensure you have access

**Warning: "Could not extract detailed storage info"**
- This is expected for some vLLM models
- BPW will default to 4.0 (typical for FP4)
- You can manually verify BPW if needed

**Perplexity seems wrong**
- Verify tokenizer matches your model
- Check that the base model architecture is correct
- Ensure `eval_len` (2048) fits in your model's context window

## Color Coding in Plots

To have your vLLM models show up with specific colors in plots, include color codes in your label:

```json
{
    "label": "vLLM [cornflowerblue] NVFP4"
}
```

Available colors:
- `green`/`greenyellow` - EXL2
- `purple`/`palevioletred` - EXL3
- `red`/`tomato` - GGUF
- `blue`/`cornflowerblue` - VPTQ
- `teal`/`lightseagreen` - QTIP
- `black`/`silver` - Other

## Example: Full Comparison Workflow

```bash
# 1. Create your vLLM model spec
cat > eval/spec/my_nvfp4_models.json << 'EOF'
[
    {
        "load_fn": "vllm",
        "fwd_fn": "vllm",
        "label": "vLLM [blue] NVFP4 Llama-3.1-8B",
        "model_dir": "/models/llama-3.1-8b-nvfp4"
    }
]
EOF

# 2. Run comparison against EXL3
cd eval
python compare_q.py \
    -d spec/wiki2_llama3.json \
    -m spec/my_nvfp4_models.json spec/llama3.1-8b-instruct_exl3.json \
    -p \
    -t "NVFP4 vs EXL3: Llama-3.1-8B-Instruct" \
    -my 15 \
    -mx 10

# 3. Results will be displayed and plotted
```

## Technical Details

### How It Works

1. **Model Loading**: Uses vLLM's `LLM` class with offline inference mode
2. **Forward Pass**: Calls `model.generate()` with `prompt_logprobs=1` to get log probabilities
3. **Perplexity Calculation**: Converts logprobs to logits format compatible with the standard perplexity calculation
4. **Storage Info**: Scans GPU memory to estimate model size and bits-per-weight

### File Structure

```
eval/
├── compare_q.py              # Main comparison script
├── compare_q_vllm.py         # vLLM integration (NEW)
├── spec/
│   ├── example_vllm_nvfp4.json   # Example vLLM spec (NEW)
│   └── wiki2_llama3.json     # Data specification
└── VLLM_USAGE.md             # This file (NEW)
```

## Contributing

If you find issues with the vLLM integration or have improvements:
1. Test thoroughly with your NVFP4 models
2. Document any edge cases
3. Submit issues or pull requests to the ExLlamaV3 repository

## References

- [vLLM Documentation](https://docs.vllm.ai/)
- [Compressed Tensors Format](https://github.com/neuralmagic/compressed-tensors)
- [NVIDIA FP4 Documentation](https://nvidia.github.io/TensorRT-LLM/reference/precision.html)


