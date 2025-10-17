"""
Custom perplexity evaluation script using vLLM's Python API.
This script properly tests QUANTIZED models (not decompressed).
"""

import torch
from vllm import LLM, SamplingParams, TokensPrompt
from datasets import load_dataset
from tqdm import tqdm
import math
import argparse


def calculate_perplexity(model_path: str, max_model_len: int = 2048, 
                        context_length: int = 2048, stride: int = 512,
                        quantization: str = "compressed-tensors",
                        tensor_parallel_size: int = 1,
                        gpu_memory_utilization: float = 0.85,
                        max_samples: int = None):
    """
    Calculate perplexity using vLLM with proper quantized inference.
    
    Args:
        model_path: Path to the model
        max_model_len: Maximum sequence length the model can handle
        context_length: Length of context for perplexity calculation
        stride: Stride for rolling window
        quantization: Quantization method (compressed-tensors, awq, etc.)
        tensor_parallel_size: Number of GPUs for tensor parallelism
        gpu_memory_utilization: GPU memory utilization fraction
        max_samples: Maximum number of samples to evaluate (None for all)
    """
    
    # Initialize vLLM with quantization
    print(f"Loading model: {model_path}")
    print(f"Quantization: {quantization}")
    print(f"Max model length: {max_model_len}")
    
    # IMPORTANT: Set max_model_len = context_length to avoid the +1 token issue
    llm = LLM(
        model=model_path,
        quantization=quantization,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=context_length,  # Set to context length, not +1
        trust_remote_code=True,
        seed=1234,
    )
    
    tokenizer = llm.get_tokenizer()
    
    # Load WikiText-2 test set
    print("Loading WikiText-2 dataset...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    
    # Tokenize the entire test set
    print("Tokenizing dataset...")
    encodings = tokenizer("\n\n".join(dataset["text"]), return_tensors="pt")
    input_ids = encodings.input_ids[0]
    
    print(f"Total tokens in dataset: {len(input_ids)}")
    
    # Calculate number of stride windows
    seq_len = len(input_ids)
    nlls = []
    prev_end_loc = 0
    
    # Create sampling params that request prompt logprobs
    # IMPORTANT: prompt_logprobs must be high enough to include actual tokens
    # If a token isn't in top-K, we can't calculate its probability!
    # Setting to None requests ALL vocabulary logprobs (expensive but necessary for accuracy)
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=1,  # We need at least 1 token generation
        prompt_logprobs=None,  # None = return all vocab logprobs (or use large number like 100)
        logprobs=None,  # Also request full logprobs for generated tokens
    )
    
    print(f"Calculating perplexity with stride {stride}...")
    
    num_windows = 0
    for begin_loc in tqdm(range(0, seq_len, stride)):
        end_loc = min(begin_loc + context_length, seq_len)
        trg_len = end_loc - prev_end_loc  # Length of target tokens
        
        # Extract the token IDs for this window
        input_ids_window = input_ids[begin_loc:end_loc].tolist()
        
        # Skip if window is too short
        if len(input_ids_window) < 2:
            break
            
        # CRITICAL: Only use context_length - 1 tokens as input
        # to leave room for the 1 token generation vLLM requires
        if len(input_ids_window) >= context_length:
            input_ids_window = input_ids_window[:context_length - 1]
            
        # Generate with vLLM using TokensPrompt
        try:
            outputs = llm.generate(
                prompts=[TokensPrompt(prompt_token_ids=input_ids_window)],
                sampling_params=sampling_params,
                use_tqdm=False,
            )
            
            output = outputs[0]
            
            # Extract prompt logprobs
            if output.prompt_logprobs is None:
                print(f"Warning: No prompt logprobs for window {num_windows}")
                continue
                
            # Calculate negative log likelihood for target tokens
            # Skip the first token (no logprob) and only use the target window
            prompt_logprobs = output.prompt_logprobs[1:]  # Skip first token
            
            if len(prompt_logprobs) < trg_len:
                trg_len = len(prompt_logprobs)
            
            # Sum the log probabilities of the actual tokens
            window_nll = 0.0
            for i, logprob_dict in enumerate(prompt_logprobs[-trg_len:]):
                if logprob_dict is None:
                    continue
                # Get the actual token at this position
                actual_token = input_ids[prev_end_loc + i].item()
                if actual_token in logprob_dict:
                    # logprob_dict[token] is a Logprob object, get its .logprob attribute
                    token_logprob = getattr(logprob_dict[actual_token], 'logprob', logprob_dict[actual_token])
                    window_nll -= token_logprob
                else:
                    print(f"Warning: Token {actual_token} not in logprobs at position {i}")
                    
            nlls.append(window_nll)
            num_windows += 1
            
        except Exception as e:
            print(f"Error processing window {num_windows}: {e}")
            continue
        
        prev_end_loc = end_loc
        
        # Check if we've reached max samples
        if max_samples and num_windows >= max_samples:
            break
            
        if end_loc == seq_len:
            break
    
    # Calculate perplexity
    if len(nlls) == 0:
        print("Error: No valid windows processed")
        return None
        
    total_nll = sum(nlls)
    num_tokens = prev_end_loc
    
    ppl = math.exp(total_nll / num_tokens)
    
    print(f"\nResults:")
    print(f"  Processed {num_windows} windows")
    print(f"  Total tokens evaluated: {num_tokens}")
    print(f"  Average NLL: {total_nll / num_tokens:.4f}")
    print(f"  Perplexity: {ppl:.4f}")
    
    return ppl


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate perplexity using vLLM with quantization")
    parser.add_argument("--model", type=str, required=True, help="Path to model")
    parser.add_argument("--context-length", type=int, default=2048, help="Context length for evaluation")
    parser.add_argument("--stride", type=int, default=512, help="Stride for rolling window")
    parser.add_argument("--quantization", type=str, default="compressed-tensors", help="Quantization method")
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85, help="GPU memory utilization")
    parser.add_argument("--max-samples", type=int, default=None, help="Max number of windows to evaluate")
    
    args = parser.parse_args()
    
    calculate_perplexity(
        model_path=args.model,
        max_model_len=args.context_length,
        context_length=args.context_length,
        stride=args.stride,
        quantization=args.quantization,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_samples=args.max_samples,
    )

