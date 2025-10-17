# Perplexity Evaluation Problem Statement for Quantized Models

## Objective

**Primary Goal**: Obtain accurate, comparable perplexity scores for compressed-tensors NVFP4 (W4A16) quantized models stored in Hugging Face format, specifically testing the *quantized* weights without decompression to FP16/BF16.

**Requirements**:
- Must test the actual quantized model (compressed-tensors format with NVFP4/W4A16)
- Perplexity must be calculated with exact log probabilities (not approximations)
- Results must be comparable to existing benchmarks (e.g., EXL2/EXL3 comparisons)
- Must use standard WikiText-2 dataset with 2048 token context length
- Must support large models (70B+ parameters) that require multiple GPUs

## Why Standard Tools Don't Work

### 1. lm-evaluation-harness (`lm-eval`)

#### With `hf` Backend
**Problem**: Automatic weight decompression to FP16
```bash
lm_eval --model hf \
  --model_args pretrained=/path/to/W4A16/model,trust_remote_code=True \
  --tasks wikitext --batch_size 1
```

- **What happens**: The `transformers` library automatically decompresses INT4 weights to FP16 when loading
- **Memory impact**: 70B W4A16 model becomes 140GB in VRAM (70B × 2 bytes), exceeding single GPU capacity
- **Result**: ❌ OOM errors OR tests decompressed model instead of quantized model
- **Verdict**: Defeats the entire purpose of testing the quantization

#### With `vllm` Backend (Direct Integration)
**Problem**: CUDA multiprocessing fork() incompatibility
```bash
lm_eval --model vllm \
  --model_args pretrained=/path/to/W4A16/model,quantization=compressed-tensors \
  --tasks wikitext
```

- **Error**: `RuntimeError: Cannot re-initialize CUDA in forked subprocess`
- **Cause**: lm-eval initializes CUDA, then vLLM tries to use fork() for multiprocessing
- **Attempted fix**: `VLLM_WORKER_MULTIPROC_METHOD=spawn` + `--enable_v1` flags
- **New error**: `AssertionError: Sampled token IDs exceed the max model length` (tokens + required output > max_model_len)
- **Verdict**: ❌ Fundamental incompatibility between lm-eval and vLLM's execution model

#### With vLLM API Server (HTTP)
**Problem**: Completions API doesn't return accurate logprobs for perplexity
```bash
# Terminal 1
vllm serve /path/to/W4A16/model --quantization compressed-tensors

# Terminal 2
lm_eval --model local-completions \
  --model_args model=/path/to/model,base_url=http://localhost:8000/v1
```

- **What happens**: vLLM's OpenAI-compatible `/v1/completions` endpoint is designed for generation, not scoring
- **Result**: Perplexity of 21.67 for a 70B model (should be ~7-9)
- **Verdict**: ❌ API not suitable for accurate perplexity measurement

### 2. ExLlamaV3's `eval/compare_q.py`

**Problem**: Framework designed for EXL2/EXL3 formats, not compressed-tensors

#### Initial Attempt
```bash
python eval/compare_q.py \
  -d eval/spec/wiki2_small_model.json \
  -m eval/spec/small_nvfp4_transformers.json
```

- **Issue 1**: Script had `logits += 1e-10` before `log_softmax`, distorting probability distribution
- **Result**: Perplexity of 389 for 8B W4A16 model (should be ~9-10)
- **Fix**: Removed the `+= 1e-10` line

#### After Fix
- **Issue 2**: Limited sampling (max_rows=20) causes high variance in results
- **Issue 3**: Uses `transformers` AutoModel which still decompresses INT4 to FP16
- **Issue 4**: No native support for compressed-tensors layer types (`CompressedLinear`)
- **Verdict**: ❌ Can run but defeats purpose by decompressing weights

### 3. Direct Hugging Face Transformers

**Problem**: Automatic quantization decompression during inference

```python
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(
    "/path/to/W4A16/model",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)
```

- **What happens**: `compressed-tensors` library's `CompressedLinear` layers decompress INT4 → FP16 during forward pass
- **Memory**: Full FP16 activations stored in VRAM during computation
- **Verdict**: ❌ Same problem as lm-eval with `hf` backend

### 4. vLLM Python API (`vllm.LLM`)

**Problem**: No API to retrieve exact log probabilities for arbitrary tokens

#### Custom Script Attempt (`eval/vllm_perplexity.py`)
```python
from vllm import LLM, SamplingParams

llm = LLM(model=model_path, quantization="compressed-tensors")
sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=1,
    prompt_logprobs=20  # Maximum allowed by vLLM
)
outputs = llm.generate(prompts=[TokensPrompt(prompt_token_ids=input_ids)], 
                       sampling_params=sampling_params)
```

**Critical Limitation**: 
- `prompt_logprobs=N` returns **only top-N most likely tokens**, not the actual ground-truth token
- vLLM has a hard-coded limit of `max=20` for prompt_logprobs
- **Result**: 90%+ of actual tokens are missing from returned logprobs
- **Example output**:
  ```
  Warning: Token 546 not in logprobs at position 301
  Warning: Token 72 not in logprobs at position 302
  ...
  Warning: Window 0 had only 4/308 valid tokens, skipping
  ```

**Why This Happens**:
- vLLM is optimized for **generation** (sampling from top-K), not **scoring** (evaluating specific sequences)
- Perplexity requires the log probability of the *actual next token*, which may not be in top-20
- No `encode()` or `score()` method exists that returns exact logprobs for given token sequences

**Verdict**: ❌ Architecturally unsuitable for perplexity evaluation

## The Fundamental Problem

### What Perplexity Calculation Requires

For each token position `i` in a sequence:
```
NLL(i) = -log P(token_i | context_{0:i-1})
Perplexity = exp(sum(NLL) / num_tokens)
```

**Critical requirement**: Need the **exact log probability of the actual token** at each position, not just the top-K predictions.

### Why Quantized Inference Engines Fail

1. **Transformers (HuggingFace)**: Decompresses quantized weights to FP16 during inference
   - ✅ Provides exact logprobs
   - ❌ Tests decompressed model, not quantized model

2. **vLLM**: Keeps weights compressed during inference
   - ✅ Tests actual quantized model
   - ❌ Only provides top-K logprobs (generation-optimized)

3. **ExLlamaV3**: Format-specific (EXL2/EXL3)
   - ✅ Tests actual quantized model
   - ❌ Doesn't support compressed-tensors/NVFP4 format

## Current Status

**No viable solution exists** to accurately measure perplexity of compressed-tensors NVFP4 quantized models while keeping weights compressed during inference.

### Attempted Workarounds (All Failed)

1. **Approximate perplexity with top-20**: Assign penalty to missing tokens
   - Result: >95% tokens missing, perplexity meaningless

2. **Increase max_model_len to avoid truncation**: Still hit assertion errors in vLLM

3. **Use V0 engine instead of V1**: Different errors, same fundamental limitation

4. **Serve vLLM separately and call via API**: Completions API gives inaccurate scores

## Potential Paths Forward (Unexplored)

1. **Feature Request to vLLM**: Add a `score()` method to `vllm.LLM` that returns exact logprobs for given token sequences
   - Would require vLLM architecture changes
   - Similar to how some engines support "loglikelihood" mode

2. **Alternative Inference Engines**: 
   - llama.cpp with perplexity mode (requires GGUF conversion)
   - TensorRT-LLM (requires TRT format conversion)
   - SGLang (may have similar limitations)

3. **Format Conversion**:
   - Convert compressed-tensors → GGUF → run perplexity with llama.cpp
   - Risk: Conversion may not preserve exact quantization behavior

4. **Manual Implementation**:
   - Write custom CUDA kernels to run compressed-tensors inference
   - Extract exact logits for perplexity calculation
   - Extremely high effort, maintenance burden

## Conclusion

Testing perplexity of compressed-tensors quantized models without weight decompression is **currently impossible** with standard tools. The ecosystem has a gap:

- **Scoring tools** (lm-eval, transformers) decompress weights
- **Quantized inference engines** (vLLM) don't expose exact logprobs
- **Format-specific tools** (ExLlamaV3) don't support compressed-tensors

This is a fundamental architectural mismatch between quantization inference optimizations and perplexity evaluation requirements.

