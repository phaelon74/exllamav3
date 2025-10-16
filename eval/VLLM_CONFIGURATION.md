# vLLM Configuration for Benchmarking

## Your Specific Model Setup

Based on your serving command, here's how to configure the benchmark for your model.

### Your Serving Parameters
```bash
vllm serve /media/fmodels/.../Behemoth-R1-123B-v2_Compressed-Tensors/NVFP4 \
  --trust-remote-code \
  --quantization compressed-tensors \
  --tensor-parallel-size 2 \
  --max-model-len 65535 \
  --gpu-memory-utilization 0.80 \
  --disable-custom-all-reduce
```

### Benchmark Configuration

The model spec file (`eval/spec/example_vllm_nvfp4.json`) now supports these parameters:

```json
[
    {
        "load_fn": "vllm",
        "fwd_fn": "vllm",
        "label": "vLLM NVFP4 Behemoth 123B",
        "model_dir": [
            "/media/fmodels/TheHouseOfTheDude/Behemoth-R1-123B-v2_Compressed-Tensors/NVFP4",
            {
                "quantization": "compressed-tensors",
                "tensor_parallel_size": 2,
                "max_model_len": 2048,
                "gpu_memory_utilization": 0.80,
                "disable_custom_all_reduce": true
            }
        ]
    }
]
```

---

## Parameter Explanation

### Critical Parameters (Match Your Serving)

| Parameter | Your Serving Value | Benchmark Value | Why Different? |
|-----------|-------------------|-----------------|----------------|
| **tensor_parallel_size** | 2 | 2 ✅ | **MUST MATCH** - Your 123B model needs 2 GPUs |
| **quantization** | compressed-tensors | compressed-tensors ✅ | **MUST MATCH** - NVFP4 format |
| **disable_custom_all_reduce** | true | true ✅ | **SHOULD MATCH** - Hardware compatibility |
| **gpu_memory_utilization** | 0.80 | 0.80 ✅ | **CAN MATCH** - Safe memory usage |
| **trust_remote_code** | true | true ✅ | **MUST MATCH** - Model requires it |

### Different Parameters (This is OK!)

| Parameter | Your Serving | Benchmark | Why? |
|-----------|-------------|-----------|------|
| **max_model_len** | 65535 | 2048 | ✅ **Eval only uses 2048 tokens**<br>No need to allocate 65K context for benchmarking |

---

## Important: Context Length

### Your Serving Setup
- `max_model_len: 65535` - Supports very long contexts for production use

### Benchmark Setup
- `max_model_len: 2048` - Evaluation sequences are only 2048 tokens

**Why this works:**
1. The **data spec** (`wiki2_llama3.json`) controls actual sequence length
2. It uses `"eval_len": 2048` - only 2048 tokens per test
3. Setting `max_model_len: 2048` saves memory and speeds up loading
4. You **can** set it to 65535 if you want, but it wastes VRAM

### Data Spec Controls Context

Look at `eval/spec/wiki2_llama3.json`:
```json
{
    "eval_len": 2048,     ← Actual sequence length used
    "eval_stride": 512,
    "max_rows": 20
}
```

This is what actually determines how much context the benchmark uses.

---

## How to Use Custom Parameters

### Method 1: Edit the JSON File (Recommended)

Edit `eval/spec/example_vllm_nvfp4.json`:

```json
{
    "model_dir": [
        "/your/model/path",
        {
            "tensor_parallel_size": 2,
            "quantization": "compressed-tensors",
            "gpu_memory_utilization": 0.80,
            "disable_custom_all_reduce": true,
            "max_model_len": 2048
        }
    ]
}
```

### Method 2: Simple Path (Uses Defaults)

If you only have 1 GPU and standard settings:
```json
{
    "model_dir": "/your/model/path"
}
```

This uses defaults:
- `tensor_parallel_size`: 1 (default)
- `gpu_memory_utilization`: 0.9
- `max_model_len`: 2048
- `trust_remote_code`: true

---

## Full Example for Your Model

### File: `eval/spec/behemoth_123b_nvfp4.json`

```json
[
    {
        "load_fn": "vllm",
        "fwd_fn": "vllm",
        "label": "Behemoth R1 123B NVFP4",
        "model_dir": [
            "/media/fmodels/TheHouseOfTheDude/Behemoth-R1-123B-v2_Compressed-Tensors/NVFP4",
            {
                "quantization": "compressed-tensors",
                "tensor_parallel_size": 2,
                "max_model_len": 2048,
                "gpu_memory_utilization": 0.80,
                "disable_custom_all_reduce": true,
                "trust_remote_code": true
            }
        ]
    }
]
```

### Run Command

```bash
python eval/compare_q.py \
    -d eval/spec/wiki2_llama3.json \
    -m eval/spec/behemoth_123b_nvfp4.json \
    -p \
    -t "Behemoth R1 123B NVFP4 Perplexity"
```

---

## Available vLLM Parameters

You can pass any vLLM `LLM()` parameter in the second element of the list:

```json
{
    "model_dir": [
        "/path/to/model",
        {
            "quantization": "compressed-tensors",
            "tensor_parallel_size": 2,
            "max_model_len": 2048,
            "gpu_memory_utilization": 0.80,
            "trust_remote_code": true,
            "disable_custom_all_reduce": true,
            "enforce_eager": false,
            "swap_space": 4,
            "dtype": "auto",
            "download_dir": "/custom/cache"
        }
    ]
}
```

See [vLLM LLM class documentation](https://docs.vllm.ai/en/latest/offline_inference/llm.html) for all options.

---

## Memory Considerations

### Your 123B Model on 2 GPUs

With `tensor_parallel_size: 2`:
- Model is split across 2 GPUs
- Each GPU holds ~half the model
- NVFP4 = ~4 bits per weight
- 123B params × 4 bits = 492 Gb = 61.5 GB
- Per GPU: ~30-35 GB (plus overhead)

### GPU Memory Utilization

```json
"gpu_memory_utilization": 0.80
```

- Leaves 20% free for KV cache and activations
- Conservative and safe
- Can increase to 0.90 if needed (but risky)

### Context Length Trade-off

| Setting | VRAM for KV Cache | Speed | Use Case |
|---------|-------------------|-------|----------|
| `max_model_len: 2048` | ~2 GB | Fast ✅ | Benchmarking (recommended) |
| `max_model_len: 8192` | ~8 GB | Medium | Long context eval |
| `max_model_len: 65535` | ~65 GB | Slow ❌ | Production (not needed for eval) |

**Recommendation**: Keep `max_model_len: 2048` for benchmarking.

---

## Troubleshooting

### Error: CUDA Out of Memory
```
torch.cuda.OutOfMemoryError: CUDA out of memory
```

**Solutions:**
1. Lower `gpu_memory_utilization` from 0.80 to 0.70
2. Reduce `max_model_len` to 1024 or 512
3. Reduce `max_rows` in data spec from 20 to 10

### Error: Tensor Parallel Size Mismatch
```
ValueError: Number of available GPUs < tensor_parallel_size
```

**Solution:** Make sure you have 2 GPUs available:
```bash
nvidia-smi  # Check GPU availability
```

### Error: Custom All-Reduce Failed
```
RuntimeError: NCCL error
```

**Solution:** Keep `disable_custom_all_reduce: true` in your config.

---

## Verification

After running, you should see:
```
Loading vLLM model from: /media/fmodels/.../NVFP4
vLLM parameters: {'model': '...', 'tensor_parallel_size': 2, ...}
INFO: Initializing an LLM engine with 2 GPUs...
Loaded vLLM model: 4.00 bpw (decoder), 4.00 bpw (head)
```

This confirms:
- ✅ Model path is correct
- ✅ 2-GPU tensor parallelism is active
- ✅ Compressed tensors quantization detected
- ✅ Model loaded successfully

---

## Quick Reference

### Your Exact Configuration

**File: `eval/spec/example_vllm_nvfp4.json`** (already configured!)
```json
{
    "model_dir": [
        "/media/fmodels/TheHouseOfTheDude/Behemoth-R1-123B-v2_Compressed-Tensors/NVFP4",
        {
            "quantization": "compressed-tensors",
            "tensor_parallel_size": 2,
            "max_model_len": 2048,
            "gpu_memory_utilization": 0.80,
            "disable_custom_all_reduce": true
        }
    ]
}
```

**Run Command:**
```bash
python eval/compare_q.py \
    -d eval/spec/wiki2_llama3.json \
    -m eval/spec/example_vllm_nvfp4.json
```

That's it! The script will now use your serving parameters. 🚀

