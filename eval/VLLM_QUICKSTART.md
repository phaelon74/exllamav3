# vLLM Quick Start Guide

## What Was Created

Three new files have been added to enable vLLM benchmarking:

1. **`eval/compare_q_vllm.py`** - vLLM integration module
2. **`eval/spec/example_vllm_nvfp4.json`** - Example model specification
3. **`eval/VLLM_USAGE.md`** - Comprehensive documentation

The main comparison script (`eval/compare_q.py`) has been updated to recognize vLLM models.

## Quick Usage

### 1. Install vLLM

```bash
pip install vllm
```

### 2. Create Your Model Spec

Edit `eval/spec/example_vllm_nvfp4.json` and change the `model_dir` to point to your NVFP4 model:

```json
[
    {
        "load_fn": "vllm",
        "fwd_fn": "vllm",
        "label": "vLLM NVFP4",
        "model_dir": "/your/path/to/nvfp4/model"
    }
]
```

### 3. Run Benchmark

```bash
cd eval

# Simple perplexity test
python compare_q.py \
    -d spec/wiki2_llama3.json \
    -m spec/example_vllm_nvfp4.json

# With visualization
python compare_q.py \
    -d spec/wiki2_llama3.json \
    -m spec/example_vllm_nvfp4.json \
    -p \
    -t "My NVFP4 Model Benchmark"
```

### 4. Compare with Other Formats

```bash
# Compare NVFP4 vs EXL3
python compare_q.py \
    -d spec/wiki2_llama3.json \
    -m spec/example_vllm_nvfp4.json spec/llama3.1-8b-instruct_exl3.json \
    -p \
    -t "NVFP4 vs EXL3"
```

## Expected Output

```
Loading vLLM model from: /your/path/to/nvfp4/model
Loaded vLLM model: 4.00 bpw (decoder), 4.00 bpw (head)
Loading dataset: wiki2
Testing: /your/path/to/nvfp4/model (vLLM NVFP4)
Evaluating: 100% |████████████████████| 20/20
Perplexity: 8.523456

------
[
    {
        "label": "vLLM NVFP4",
        "layer_bpw": 4.0,
        "head_bpw": 4.0,
        "vram_gb": 3.45,
        "ppl": 8.523456
    }
]
```

## Important Notes

- **Tokenizer**: Ensure the `tokenizer_dir` in your data spec matches your model
- **Memory**: vLLM needs significant VRAM; reduce `max_rows` if you run out
- **BPW Values**: May be approximate (default 4.0 for FP4) due to vLLM limitations
- **Batch Size**: Currently limited to 1 due to vLLM API constraints

## Troubleshooting

**Error: CUDA out of memory**
- Edit your data spec and reduce `max_rows` from 20 to 10

**Error: Model not found**
- Check `model_dir` path is correct
- For HuggingFace models, use the full model ID

**Perplexity seems off**
- Verify tokenizer matches your model family
- Check model architecture is correct

## Full Documentation

See `eval/VLLM_USAGE.md` for comprehensive documentation including:
- Advanced options
- Color coding for plots
- Technical details
- Complete examples

