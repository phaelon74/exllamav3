# Quick Start: NVFP4 Perplexity Benchmark (Exact Scores)

## TL;DR - Run This Command

```bash
cd ~/vllm/exllamav3

# Activate your venv with transformers installed
source ~/vllm/venv/bin/activate

# Quick test (5 rows, ~5-10 minutes)
python eval/compare_q.py \
    -d eval/spec/wiki2_behemoth_fast.json \
    -m eval/spec/behemoth_nvfp4_transformers.json

# Full benchmark (20 rows, ~20-30 minutes)
python eval/compare_q.py \
    -d eval/spec/wiki2_behemoth.json \
    -m eval/spec/behemoth_nvfp4_transformers.json
```

## What You'll Get

```
Loading model with tensor parallelism: /media/fmodels/.../NVFP4
Parameters: {'device_map': 'auto', 'torch_dtype': torch.bfloat16, ...}

Loaded transformers model: 4.00 bpw (decoder), 4.00 bpw (head)
Testing: [...] (NVFP4 (Transformers))

Evaluating: 100%|████████████████████| 20/20 [XX:XX<00:00]

Perplexity: 7.234567  ← Your exact, comparable score!
```

## Why This Works

✅ **Exact Perplexity**: Full vocabulary logits, not approximate top-20
✅ **Comparable**: Directly matches ExLlamaV3 benchmark methodology
✅ **Tensor Parallel**: Uses both your GPUs automatically
✅ **Compressed-Tensors**: Native NVFP4 support via Transformers

❌ **Not vLLM**: vLLM can't provide exact perplexity (see PERPLEXITY_COMPARISON_ANALYSIS.md)

## Files Created

1. **`eval/compare_q_transformers.py`** (modified)
   - Added `load_transformers_tensor_parallel()` function
   - Added `fwd_transformers_auto()` function
   - Supports multi-GPU compressed-tensors models

2. **`eval/compare_q.py`** (modified)
   - Registered `"transformers_tp"` load function
   - Registered `"transformers_auto"` forward function

3. **`eval/spec/behemoth_nvfp4_transformers.json`** (new)
   - Your model configuration
   - Uses `"transformers_tp"` backend
   - Points to your NVFP4 model directory

4. **`eval/spec/wiki2_behemoth.json`** (already existed)
   - WikiText-2 dataset configuration
   - Correct tokenizer path for your model

5. **`eval/spec/wiki2_behemoth_fast.json`** (already existed)
   - Fast testing version (5 rows instead of 20)

## Requirements Check

Before running, ensure:

```bash
# Check transformers version (need >= 4.46.0 for compressed-tensors)
python -c "import transformers; print(transformers.__version__)"

# Check compressed-tensors is installed
python -c "import compressed_tensors; print('OK')"

# If missing, install:
pip install transformers>=4.46.0 compressed-tensors
```

## Compare Against Multiple Models

```bash
# Compare NVFP4 vs EXL3 quantizations
python eval/compare_q.py \
    -d eval/spec/wiki2_behemoth.json \
    -m eval/spec/behemoth_nvfp4_transformers.json \
    -m eval/spec/your_exl3_4bpw.json \
    -m eval/spec/your_exl3_5bpw.json \
    --plot
```

## Troubleshooting

### OOM Error
Reduce context length in data spec:
```json
{
    "eval_len": 1024,
    "eval_stride": 256
}
```

### Model Won't Load
Check directory:
```bash
ls /media/fmodels/TheHouseOfTheDude/Behemoth-R1-123B-v2_Compressed-Tensors/NVFP4/
# Should have: config.json, *.safetensors, tokenizer*
```

### Too Slow
Use fast spec:
```bash
python eval/compare_q.py \
    -d eval/spec/wiki2_behemoth_fast.json \
    -m eval/spec/behemoth_nvfp4_transformers.json
```

## Expected Performance

| Dataset | Rows | Time (2x GPU) | Result |
|---------|------|---------------|--------|
| Fast | 5 | ~5-10 min | Quick sanity check |
| Full | 20 | ~20-30 min | Benchmark quality |

## Next Steps

1. ✅ Run quick test first (`wiki2_behemoth_fast.json`)
2. ✅ Verify perplexity is reasonable (5-15 range)
3. ✅ Run full benchmark (`wiki2_behemoth.json`)
4. ✅ Compare to other quantization methods
5. ✅ Share results!

## Documentation

- **Full Guide**: `eval/TRANSFORMERS_TENSOR_PARALLEL_GUIDE.md`
- **Technical Analysis**: `PERPLEXITY_COMPARISON_ANALYSIS.md`
- **vLLM Integration**: `eval/VLLM_USAGE.md` (for reference, not perplexity)

## Support

Questions? Check:
1. Troubleshooting section above
2. Full guide in `TRANSFORMERS_TENSOR_PARALLEL_GUIDE.md`
3. Analysis doc in `PERPLEXITY_COMPARISON_ANALYSIS.md`


