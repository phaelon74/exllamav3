# NVFP4 OOM (Out of Memory) Solutions

## The Problem

Your 123B NVFP4 model ran out of memory on GPU 0 (94.52 GB / 94.97 GB used) even though you have 2 GPUs. This happens because:

1. **Compressed-tensors decompression overhead**: NVFP4 weights are stored compressed (~4-bit) but must be decompressed to FP16/BF16 during forward pass, creating temporary tensors
2. **Uneven distribution**: `device_map="auto"` doesn't account for decompression overhead
3. **Large context length**: 2048 tokens requires significant activation memory

## Solutions (Try in Order)

### ✅ Solution 1: Use Reduced Context Length (Easiest)

I've created files that use 1024 tokens instead of 2048:

```bash
# Make script executable
chmod +x eval/run_nvfp4_benchmark.sh

# Run with automatic memory management
./eval/run_nvfp4_benchmark.sh
```

**What this does:**
- Sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to reduce fragmentation
- Uses 1024 token context (still valid perplexity measurement)
- Uses the updated spec with manual memory limits

**Pros:** Should work immediately
**Cons:** Slightly different context length than standard benchmarks (but still comparable)

---

### ✅ Solution 2: Manual Memory Distribution

If you want to keep 2048 context, force more aggressive GPU 1 usage:

**Edit `eval/spec/behemoth_nvfp4_transformers.json`:**

```json
{
    "max_memory": {
        "0": "30GiB",    // Even less for GPU 0
        "1": "92GiB"     // Almost all on GPU 1
    }
}
```

**Then run:**
```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python eval/compare_q.py \
    -d eval/spec/wiki2_behemoth_fast.json \
    -m eval/spec/behemoth_nvfp4_transformers.json
```

---

### ✅ Solution 3: Use Float16 Instead of BFloat16

BFloat16 and Float16 use the same memory, but Float16 might allocate differently:

**Edit `eval/spec/behemoth_nvfp4_transformers.json`:**

```json
{
    "trust_remote_code": true,
    "dtype": "float16",   // Add this line
    "max_memory": {
        "0": "38GiB",
        "1": "90GiB"
    }
}
```

---

### ✅ Solution 4: CPU Offload (Slowest but Most Reliable)

Offload some layers to CPU RAM:

**Edit `eval/spec/behemoth_nvfp4_transformers.json`:**

```json
{
    "trust_remote_code": true,
    "max_memory": {
        "0": "35GiB",
        "1": "85GiB",
        "cpu": "50GiB"   // Offload overflow to CPU
    }
}
```

**Pros:** Will definitely work
**Cons:** Much slower due to CPU-GPU transfers

---

### ✅ Solution 5: Smallest Context (Ultra Safe)

For testing only, use very small context:

**Edit `eval/spec/wiki2_behemoth_short.json`:**

```json
{
    "eval_len": 512,      // Very short context
    "eval_stride": 128,
    "max_rows": 3         // Just 3 samples for testing
}
```

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python eval/compare_q.py \
    -d eval/spec/wiki2_behemoth_short.json \
    -m eval/spec/behemoth_nvfp4_transformers.json
```

---

## Understanding Your Error

```
GPU 0 has a total capacity of 94.97 GiB of which 452.56 MiB is free.
Including non-PyTorch memory, this process has 94.52 GiB memory in use.
Of the allocated memory 92.64 GiB is allocated by PyTorch
```

**Translation:**
- GPU 0: 94.52 GB used / 94.97 GB total (99.5% full!)
- GPU 1: Probably underutilized
- The model layers are mostly on GPU 0
- During forward pass, NVFP4 decompression tried to allocate 672 MB more on GPU 0
- Boom! 💥

## Current Configuration

**What I've already set up:**

1. **`eval/spec/behemoth_nvfp4_transformers.json`** - Updated with memory limits:
   ```json
   "max_memory": {
       "0": "38GiB",  // Leave headroom for decompression
       "1": "90GiB"   // Use more of GPU 1
   }
   ```

2. **`eval/spec/wiki2_behemoth_short.json`** - Reduced context:
   ```json
   "eval_len": 1024,     // Half the context
   "eval_stride": 256,
   "max_rows": 5
   ```

3. **`eval/run_nvfp4_benchmark.sh`** - Script with proper env vars

## Recommended Approach

**Start here:**

```bash
# Try Solution 1 first (reduced context)
chmod +x eval/run_nvfp4_benchmark.sh
./eval/run_nvfp4_benchmark.sh
```

**If that still fails:**

```bash
# Edit behemoth_nvfp4_transformers.json to use Solution 2 (more aggressive limits)
# Change max_memory "0": "30GiB"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python eval/compare_q.py \
    -d eval/spec/wiki2_behemoth_short.json \
    -m eval/spec/behemoth_nvfp4_transformers.json
```

**If that STILL fails:**

```bash
# Add CPU offload (Solution 4)
# Edit behemoth_nvfp4_transformers.json to add "cpu": "50GiB" to max_memory
```

## Why This Happens with NVFP4

NVFP4 is a **compressed** format:

1. **At rest**: Weights stored as ~4-bit integers → Small memory footprint
2. **During forward**: Weights decompressed to FP16 → Temporary 16-bit tensors created
3. **Peak memory**: Compressed weights + decompressed weights + activations + gradients

This "decompression tax" isn't accounted for by `device_map="auto"`, so GPU 0 runs out of headroom.

## Alternative: Just Use vLLM for Serving

If perplexity benchmarking continues to be problematic, remember:

- **For exact perplexity**: Use smaller models or more GPUs with Transformers
- **For serving/throughput**: Use vLLM (it handles this better)
- **For downstream tasks**: Use LM Evaluation Harness with vLLM

The Transformers backend is great for exact perplexity on models that fit comfortably in VRAM, but NVFP4's decompression overhead makes it challenging for 123B models on 2 GPUs.

## Expected Memory Usage (Rough Estimates)

| Component | GPU 0 | GPU 1 | Total |
|-----------|-------|-------|-------|
| Model weights (compressed) | ~25 GB | ~35 GB | ~60 GB |
| Decompression buffers | ~10 GB | ~5 GB | ~15 GB |
| Activations (2048 ctx) | ~8 GB | ~2 GB | ~10 GB |
| KV cache | ~3 GB | ~1 GB | ~4 GB |
| Overhead | ~2 GB | ~2 GB | ~4 GB |
| **Total** | **~48 GB** | **~45 GB** | **~93 GB** |

With `max_memory` limits of `38GiB` and `90GiB`, the model should fit if we reduce context to 1024.

## Need More Help?

Check the terminal output when running to see:
1. Which GPU each layer is assigned to
2. Current memory usage during load
3. Where the OOM happens (load vs. forward)

This will help diagnose if we need even more aggressive settings.


