# Transformers Tensor Parallel Perplexity Benchmarking Guide

## Overview

This guide explains how to benchmark your NVFP4 (compressed-tensors) model using the Transformers backend with tensor parallelism to get **exact, comparable perplexity scores** that match ExLlamaV3 benchmarks.

## Why Use This Instead of vLLM?

| Feature | Transformers (This Method) | vLLM |
|---------|---------------------------|------|
| **Perplexity Accuracy** | ⭐⭐⭐⭐⭐ Exact (full vocab logits) | ⭐⭐ Approximate (top-20 only) |
| **Comparability** | ✅ Directly comparable to ExLlamaV3 | ❌ Systematically inflated scores |
| **Speed** | ⭐⭐⭐ Slower | ⭐⭐⭐⭐⭐ Faster |
| **Multi-GPU Support** | ✅ Yes (device_map="auto") | ✅ Yes (tensor_parallel_size) |
| **Compressed-Tensors** | ✅ Native support | ✅ Native support |

**Bottom Line:** For perplexity benchmarks that match ExLlamaV3 standards, use Transformers. For serving/throughput, use vLLM.

## What Was Added

### New Functions in `compare_q_transformers.py`

1. **`load_transformers_tensor_parallel(model_dir)`**
   - Loads models with `device_map="auto"` for multi-GPU distribution
   - Supports both simple string path and `[path, kwargs_dict]` format
   - Defaults: `bfloat16`, `trust_remote_code=True`, `low_cpu_mem_usage=True`

2. **`fwd_transformers_auto(model_instance, input_ids)`**
   - Forward pass that doesn't hardcode device placement
   - Automatically detects the correct device for input tensors
   - Compatible with `device_map="auto"` models

### Registration in `compare_q.py`

- `load_fns["transformers_tp"]` → `load_transformers_tensor_parallel`
- `fwd_fns["transformers_auto"]` → `fwd_transformers_auto`

## Usage

### Quick Start

```bash
# From the exllamav3 directory
python eval/compare_q.py \
    -d eval/spec/wiki2_behemoth.json \
    -m eval/spec/behemoth_nvfp4_transformers.json
```

### What This Does

1. **Loads** your NVFP4 model using Transformers with tensor parallelism
2. **Distributes** model layers across your 2 GPUs automatically
3. **Evaluates** perplexity on WikiText-2 dataset (2048 token context)
4. **Reports** exact perplexity score comparable to ExLlamaV3 results

### Expected Output

```
Loading model with tensor parallelism: /media/fmodels/TheHouseOfTheDude/Behemoth-R1-123B-v2_Compressed-Tensors/NVFP4
Parameters: {'device_map': 'auto', 'torch_dtype': torch.bfloat16, 'trust_remote_code': True, 'low_cpu_mem_usage': True}

Loading checkpoint shards: 100%|████████████████████| X/X [00:XX<00:00]

Loaded transformers model: 4.00 bpw (decoder), 4.00 bpw (head)
Testing: [...] (NVFP4 (Transformers))

Evaluating: 100%|█████████████████████████████████| 20/20 [XX:XX<00:00]

Perplexity: 7.234567  ← This is your EXACT perplexity score!
```

## Configuration Options

### Basic Configuration (Default)

The included `eval/spec/behemoth_nvfp4_transformers.json` uses sensible defaults:

```json
{
    "load_fn": "transformers_tp",
    "fwd_fn": "transformers_auto",
    "label": "NVFP4 (Transformers)",
    "model_dir": [
        "/media/fmodels/TheHouseOfTheDude/Behemoth-R1-123B-v2_Compressed-Tensors/NVFP4",
        {
            "torch_dtype": "bfloat16",
            "trust_remote_code": true
        }
    ]
}
```

### Advanced Configuration

You can add more Transformers parameters:

```json
{
    "load_fn": "transformers_tp",
    "fwd_fn": "transformers_auto",
    "label": "NVFP4 (Transformers FP16)",
    "model_dir": [
        "/path/to/your/model",
        {
            "torch_dtype": "float16",         // Use FP16 instead of BF16
            "trust_remote_code": true,
            "low_cpu_mem_usage": true,
            "device_map": "balanced",         // Alternative to "auto"
            "max_memory": {                   // Manual memory constraints
                0: "39GB",
                1: "39GB"
            }
        }
    ]
}
```

### Supported `torch_dtype` Values

- `"bfloat16"` (default) - Best compatibility
- `"float16"` - Slightly faster on older GPUs
- `"auto"` - Let Transformers decide

### Supported `device_map` Values

- `"auto"` (default) - Automatic distribution
- `"balanced"` - Balance memory across GPUs
- `"balanced_low_0"` - Keep GPU 0 free
- `{"layer_name": device}` - Manual mapping (advanced)

## Testing on Smaller Dataset (Faster)

For quick testing, use the fast data spec:

```bash
python eval/compare_q.py \
    -d eval/spec/wiki2_behemoth_fast.json \
    -m eval/spec/behemoth_nvfp4_transformers.json
```

This uses only 5 rows instead of 20, taking ~5-10 minutes instead of 20-30 minutes.

## Comparing Against Other Quantizations

### ExLlamaV3 Quantizations

Compare your NVFP4 against EXL3 quants:

```bash
python eval/compare_q.py \
    -d eval/spec/wiki2_behemoth.json \
    -m eval/spec/behemoth_nvfp4_transformers.json \
    -m eval/spec/behemoth_exl3_4bpw.json \
    -m eval/spec/behemoth_exl3_5bpw.json
```

### Other Transformers-Compatible Formats

```json
// GPTQ
{
    "load_fn": "transformers_auto",
    "fwd_fn": "transformers",
    "label": "GPTQ 4-bit",
    "model_dir": "/path/to/gptq/model"
}

// AWQ
{
    "load_fn": "transformers_auto",
    "fwd_fn": "transformers",
    "label": "AWQ 4-bit",
    "model_dir": "/path/to/awq/model"
}

// Unquantized (if you have VRAM!)
{
    "load_fn": "transformers_auto_bf16",
    "fwd_fn": "transformers",
    "label": "BF16 (Unquantized)",
    "model_dir": "/path/to/base/model"
}
```

## Troubleshooting

### Out of Memory (OOM)

If you get OOM errors:

1. **Reduce batch size** in the data spec:
   ```json
   {
       "eval_len": 1024,  // Reduce from 2048
       "eval_stride": 256  // Reduce from 512
   }
   ```

2. **Use explicit memory limits**:
   ```json
   {
       "model_dir": [
           "/path/to/model",
           {
               "max_memory": {
                   0: "35GB",  // Leave 5GB free on each GPU
                   1: "35GB"
               }
           }
       ]
   }
   ```

3. **Use FP16 instead of BF16**:
   ```json
   {
       "torch_dtype": "float16"
   }
   ```

### Model Not Loading

If model loading fails:

1. **Check compressed-tensors installation**:
   ```bash
   pip install compressed-tensors
   ```

2. **Verify transformers version**:
   ```bash
   pip install transformers>=4.46.0
   ```

3. **Check model directory**:
   ```bash
   ls /media/fmodels/TheHouseOfTheDude/Behemoth-R1-123B-v2_Compressed-Tensors/NVFP4/
   # Should contain: config.json, *.safetensors, tokenizer files
   ```

### Slow Performance

This is expected! Transformers is slower than vLLM because:
- Full vocabulary logits computation (not just top-k)
- No optimized CUDA kernels for generation
- More overhead per forward pass

**Trade-off:** 2-3x slower, but exact perplexity scores.

For a 123B model on 2 GPUs:
- **Fast (5 rows)**: ~5-10 minutes
- **Full (20 rows)**: ~20-30 minutes

## Understanding the Results

### Perplexity Score

```
Perplexity: 7.234567
```

**Lower is better.** Typical ranges:
- Excellent: 5-7
- Good: 7-10
- OK: 10-15
- Poor: 15+

### Comparison to ExLlamaV3

Your NVFP4 score is **directly comparable** to:
- EXL2 quantizations
- EXL3 quantizations
- GGUF quantizations
- Any other results from `compare_q.py`

**It is NOT comparable to:**
- vLLM perplexity scores (those will be higher due to approximation)
- Different datasets (wikitext vs. lambada)
- Different context lengths (2048 vs. 4096)

### BPW (Bits Per Weight)

```
Loaded transformers model: 4.00 bpw (decoder), 4.00 bpw (head)
```

This confirms your model is truly 4-bit quantized.

### VRAM Usage

The script automatically reports VRAM usage. For 123B @ 4-bit on 2 GPUs:
- Expected: ~60-70GB total (~30-35GB per GPU)

## Integration with ExLlamaV3 Benchmarks

### Creating a Benchmark Chart

After running multiple models:

```bash
python eval/compare_q.py \
    -d eval/spec/wiki2_behemoth.json \
    -m eval/spec/behemoth_nvfp4_transformers.json \
    -m eval/spec/behemoth_exl3_4bpw.json \
    -m eval/spec/behemoth_exl3_5bpw.json \
    --plot
```

This generates a chart showing PPL vs. BPW for all models.

### Sharing Results

Results are cached in `.cache/compare_q/` and can be shared:

```bash
# Results file
cat .cache/compare_q/results_<hash>.json
```

## Technical Details

### How It Works

1. **Load Phase** (`load_transformers_tensor_parallel`):
   - Calls `AutoModelForCausalLM.from_pretrained()` with `device_map="auto"`
   - Transformers automatically splits model across available GPUs
   - Respects compressed-tensors quantization config

2. **Forward Phase** (`fwd_transformers_auto`):
   - Detects input device from model's embedding layer
   - Moves input_ids to correct device
   - Model automatically handles cross-GPU communication
   - Returns **full vocabulary logits** (not just top-k)

3. **Perplexity Calculation** (`compare_q.py`):
   - Computes softmax over all logits: `log_probs = F.log_softmax(logits, dim=-1)`
   - Extracts exact logprob for actual next token: `target_log_probs = log_probs.gather(-1, target_ids)`
   - Calculates: `perplexity = exp(-mean(target_log_probs))`

### Why This Matches ExLlamaV3

Both Transformers and ExLlamaV3 backends:
1. Return full vocabulary logits
2. Use identical perplexity formula
3. Operate on the same tokenized data
4. Have access to exact logprobs for all tokens

The only difference is inference speed, not accuracy.

## FAQ

**Q: Will this work on a single GPU?**
A: Yes, but a 123B model requires ~60GB VRAM at 4-bit. You'll need an A100 80GB or H100.

**Q: Can I use this with other quantization formats?**
A: Yes! Works with GPTQ, AWQ, AQLM, GGUF (via transformers), and unquantized models.

**Q: Is this faster than vLLM?**
A: No. Transformers is 2-3x slower. But it's the only way to get exact perplexity.

**Q: Can I use this for downstream tasks (MMLU, etc.)?**
A: Better to use LM Evaluation Harness with vLLM for that. This is optimized for perplexity.

**Q: Why not just increase vLLM's logprobs limit?**
A: Even if increased to 100, it still won't capture all tokens. The approximation error remains.

**Q: Does this support Flash Attention?**
A: Yes, if your model config enables it and you have flash-attn installed.

## Next Steps

1. **Run the benchmark** with the provided command
2. **Note your perplexity score**
3. **Compare** to other quantization methods
4. **Share** your results on the ExLlamaV3 Discord/GitHub!

## Support

If you encounter issues:
1. Check this guide's Troubleshooting section
2. Review `PERPLEXITY_COMPARISON_ANALYSIS.md` for technical background
3. Open an issue on the ExLlamaV3 GitHub
4. Ask in the EleutherAI Discord #lm-thunderdome channel

