# Perplexity Score Comparison: ExLlamaV3 vs vLLM

## Executive Summary

**Bottom Line:** You **cannot** get directly comparable perplexity scores between ExLlamaV3 and vLLM due to fundamental architectural differences in how they provide log probabilities.

## The Problem: vLLM's Logprob Limitation

### What vLLM Does
- **Hard limit on logprobs per token**: Your version limits to 20 logprobs maximum
- **Returns only top-k tokens**: You get the 20 most likely tokens + their logprobs
- **Missing token problem**: If the actual next token isn't in the top-20, its exact logprob is unavailable

### What ExLlamaV3 Does
- **Full vocabulary logits**: Returns raw logits for ALL ~128k tokens in the vocabulary
- **Exact logprobs**: Can compute exact log probability for every actual next token
- **No approximation needed**: Always has the real value

## How Perplexity Calculation Works

### Standard Formula (ExLlamaV3)
```python
# Lines 189-207 in eval/compare_q.py
logits = model_forward(input_ids)
log_probs = F.log_softmax(logits, dim=-1)  # Get logprobs for ALL tokens
target_log_probs = log_probs.gather(-1, target_ids)  # Extract exact logprob for actual next token
mean_log_prob = target_log_probs.mean()
perplexity = math.exp(-mean_log_prob)
```

### vLLM Workaround (Our Implementation)
```python
# What we had to do in compare_q_vllm.py
outputs = vllm_model.generate(prompt, SamplingParams(prompt_logprobs=20))  # Only get top-20
prompt_logprobs = outputs[0].prompt_logprobs  # Dict with at most 20 tokens per position

# If actual next token is in top-20: use its exact logprob
# If NOT in top-20: must approximate (e.g., min_logprob - 2.0)
```

## Impact on Scores

### When Scores ARE Comparable
✅ **High-quality models** where actual next tokens are almost always in top-20
- Most tokens will be in the top-20 predictions
- Few approximations needed
- Perplexity scores will be *close* (but not identical)

### When Scores ARE NOT Comparable
❌ **Lower-quality or highly-quantized models** where predictions are worse
- Many actual next tokens fall outside top-20
- Frequent approximations needed
- Perplexity scores will be **significantly inflated** (worse)
- This is exactly the `OverflowError` we encountered

## What LM Evaluation Harness Does

### Their Approach
- Uses `loglikelihood_rolling` method for perplexity
- **Also faces the same vLLM limitation**
- Likely uses similar approximation strategies
- Documentation from EleutherAI/lm-evaluation-harness:
  - Wikitext task: `output_type: loglikelihood_rolling`
  - Metrics: `word_perplexity`, `byte_perplexity`, `bits_per_byte`

### Key Insight
LM Evaluation Harness with vLLM backend will have **the same limitation** we're experiencing. They may:
1. Have better error handling
2. Use more sophisticated approximation
3. Simply document the limitation
4. Return "N/A" for perplexity with vLLM

## Research Findings

### From GitHub Issues (vllm-project/vllm#5299)
> "Cannot request more than 5 logprobs" (version 0.4.3)
> Your version allows 20, but the fundamental issue remains

### From Community Reports
- Users report perplexity scores with vLLM being "orders of magnitude higher"
- Recommendation: Use alternative frameworks for accurate perplexity
- vLLM is optimized for **generation**, not **evaluation**

## Three Options for You

### Option 1: Accept Approximate Perplexity from vLLM ⚠️
**Pros:**
- Works with vLLM infrastructure you already have
- Can benchmark NVFP4 model
- Fast inference

**Cons:**
- Scores NOT comparable to ExLlamaV3 benchmarks
- Will be systematically higher (worse) than true perplexity
- Magnitude of error depends on model quality

**Use Case:** Relative comparisons between different vLLM models

### Option 2: Use Transformers Backend for NVFP4 ✅ RECOMMENDED
**Pros:**
- Get exact perplexity scores
- Directly comparable to ExLlamaV3 results
- Can use existing `compare_q.py` framework

**Implementation:**
```python
# In eval/spec/nvfp4_transformers.json
[
    {
        "load_fn": "transformers",
        "fwd_fn": "transformers", 
        "label": "NVFP4 (HF)",
        "model_dir": "/media/fmodels/TheHouseOfTheDude/Behemoth-R1-123B-v2_Compressed-Tensors/NVFP4",
        "dtype": "float16"  # or "bfloat16"
    }
]
```

**Cons:**
- Slower than vLLM
- May need more VRAM
- No tensor parallelism support in compare_q_transformers.py (needs modification)

### Option 3: Benchmark Other Metrics with vLLM ✅
**Instead of perplexity, measure:**
- **Throughput**: Tokens per second
- **Latency**: Time per request
- **VRAM usage**: Memory footprint
- **Quality**: Downstream task performance (accuracy, F1, etc.)

**Use LM Evaluation Harness for:**
- MMLU, HellaSwag, ARC, GSM8K, etc.
- These use `loglikelihood` not `loglikelihood_rolling`
- Work fine with vLLM's top-k limitation

## Detailed Comparison Table

| Aspect | ExLlamaV3 | vLLM (Our Integration) | LM Eval Harness + vLLM | Transformers |
|--------|-----------|------------------------|------------------------|--------------|
| Perplexity Accuracy | ⭐⭐⭐⭐⭐ Exact | ⭐⭐ Approximate | ⭐⭐ Approximate | ⭐⭐⭐⭐⭐ Exact |
| Speed | ⭐⭐⭐⭐ Fast | ⭐⭐⭐⭐⭐ Fastest | ⭐⭐⭐⭐⭐ Fastest | ⭐⭐⭐ Slow |
| VRAM Efficiency | ⭐⭐⭐⭐ Good | ⭐⭐⭐⭐⭐ Best | ⭐⭐⭐⭐⭐ Best | ⭐⭐⭐ OK |
| Tensor Parallel | ✅ Yes | ✅ Yes | ✅ Yes | ❌ No (in compare_q) |
| Task Support | EXL quantization focus | All tasks | All tasks | All tasks |
| Score Comparability | Baseline | ❌ Not comparable | ❌ Not comparable | ✅ Comparable |

## My Recommendation

### For Getting Comparable Perplexity Scores

**1. Short-term (Immediate):**
Use **Transformers backend** in the existing `compare_q.py` framework:
- Modify `compare_q_transformers.py` to support tensor parallelism (if needed for 123B model)
- Run perplexity evaluation on same dataset (wikitext-2)
- Get exact scores comparable to other ExLlamaV3 quantizations

**2. Medium-term (If you need throughput too):**
Use **LM Evaluation Harness** for comprehensive benchmarking:
- Run downstream tasks (not perplexity) with vLLM backend
- Compare MMLU, HellaSwag, etc. scores
- These work fine with vLLM's limitations

**3. Long-term (Best of both worlds):**
Create two benchmark reports:
- **Perplexity**: Use Transformers or ExLlamaV3-compatible backends
- **Throughput/Quality**: Use vLLM for downstream tasks

### What NOT To Do

❌ Don't try to "fix" the vLLM perplexity approximation
- The limitation is architectural in vLLM
- It's designed for serving, not evaluation
- Any fix will still be an approximation

❌ Don't compare vLLM perplexity to ExLlamaV3 perplexity
- Apples to oranges comparison
- Will be misleading

## Technical Deep Dive: Why This Matters

### Information Loss
```
ExLlamaV3: 128,000 logprobs → Pick exact target → PPL = exp(-mean)
vLLM:      20 logprobs → Approximate target → PPL = exp(-mean_with_approx)
```

The approximation error accumulates across thousands of tokens, leading to:
- **Bias**: Systematically higher perplexity
- **Variance**: Error magnitude depends on model quality
- **Non-comparability**: Different models affected differently

### Example Scenario
```
Token position 1523 in wikitext-2:
  Actual next token: "the" (id: 279)
  
ExLlamaV3:
  logprob(279) = -0.35  ← Exact value from full distribution
  
vLLM top-20:
  {18: -0.12, 264: -0.45, ..., 1047: -3.21}
  Token 279 NOT in top-20!
  
  Approximation: logprob(279) ≈ -3.21 - 2.0 = -5.21  ← Way off!
  
Impact on final perplexity:
  ExLlamaV3: Small contribution (exp(0.35) = 1.42)
  vLLM: Large contribution (exp(5.21) = 183.4)  ← Inflates PPL
```

## Conclusion

**Your Goal:** Get perplexity scores comparable to ExLlamaV3 benchmarks

**The Reality:** vLLM cannot provide this due to architectural limitations

**The Solution:** Use Transformers backend (Option 2) for perplexity evaluation

**Next Steps:**
1. I can modify `compare_q_transformers.py` to support tensor parallelism for your 123B model
2. Create a spec file for your NVFP4 model using Transformers backend
3. Run the perplexity benchmark and get exact, comparable scores

**Alternative:** Accept that perplexity isn't the right metric for vLLM, and focus on:
- Downstream task accuracy (MMLU, etc.)
- Throughput benchmarks
- Latency measurements

Would you like me to proceed with Option 2 (Transformers backend modification)?


