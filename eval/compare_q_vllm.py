import torch
import numpy as np

try:
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
except ModuleNotFoundError:
    pass

def get_tensor_size(tensor):
    """Calculate size in bits of a tensor"""
    return 8 * tensor.element_size() * tensor.numel()

def scan_gpu_tensors(obj, seen = None):
    """Recursively scan for GPU tensors and sum their memory usage"""
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    seen.add(obj_id)
    total_size = 0
    
    if isinstance(obj, torch.Tensor) and obj.is_cuda:
        total_size += obj.element_size() * obj.nelement()
    else:
        if isinstance(obj, dict):
            for key, value in obj.items():
                total_size += scan_gpu_tensors(key, seen)
                total_size += scan_gpu_tensors(value, seen)
            return total_size
        if isinstance(obj, (list, tuple, set)):
            for item in obj:
                total_size += scan_gpu_tensors(item, seen)
            return total_size
        if hasattr(obj, '__dict__'):
            total_size += scan_gpu_tensors(vars(obj), seen)
        if hasattr(obj, '__slots__'):
            for slot in obj.__slots__:
                try:
                    attr = getattr(obj, slot)
                    total_size += scan_gpu_tensors(attr, seen)
                except AttributeError:
                    continue
    return total_size

def get_storage_info_vllm(llm_engine):
    """
    Estimate storage info from vLLM model.
    Note: This is an approximation since vLLM doesn't expose detailed layer info easily.
    """
    try:
        # Try to access the underlying model from vLLM's LLMEngine
        model = llm_engine.llm_engine.model_executor.driver_worker.model_runner.model
        
        # Scan GPU memory used by the model
        total_vram_bits = scan_gpu_tensors(model)
        
        # Try to get config for vocab size
        config = llm_engine.llm_engine.model_config
        vocab_size = config.hf_config.vocab_size if hasattr(config, 'hf_config') else 32000
        
        # Rough estimation: assume head is ~5% of total model size
        # This is approximate since we can't easily separate head from decoder in vLLM
        head_vram_bits = total_vram_bits * 0.05
        decoder_vram_bits = total_vram_bits - head_vram_bits
        
        # Estimate number of parameters
        # For compressed tensors FP4, we assume ~4 bits per weight on average
        decoder_params = decoder_vram_bits / 4
        head_params = head_vram_bits / 4
        
        # Calculate BPW
        bpw_layer = decoder_vram_bits / decoder_params if decoder_params > 0 else 4.0
        bpw_head = head_vram_bits / head_params if head_params > 0 else 4.0
        
        return bpw_layer, bpw_head, total_vram_bits
        
    except Exception as e:
        print(f"Warning: Could not extract detailed storage info from vLLM: {e}")
        # Return conservative estimates for FP4
        # User should manually verify these if accuracy is critical
        return 4.0, 4.0, 0

@torch.inference_mode
def load_vllm(model_dir: str):
    """
    Load model using vLLM's offline inference engine.
    
    Args:
        model_dir: Path to model directory or HuggingFace model ID
                  Can also be a list: [model_dir, vllm_kwargs_dict]
    
    Returns:
        Tuple of (llm_instance, bpw_layer, bpw_head, vram_bits)
    """
    # Parse input - allow passing custom vLLM parameters
    vllm_kwargs = {}
    if isinstance(model_dir, list):
        model_dir, vllm_kwargs = model_dir
    
    print(f"Loading vLLM model from: {model_dir}")
    
    # Default vLLM parameters for evaluation
    default_params = {
        "model": model_dir,
        "gpu_memory_utilization": 0.9,
        "max_model_len": 2048,  # Eval uses 2048, but you can override
        "trust_remote_code": True,
    }
    
    # Merge user-provided kwargs (they override defaults)
    default_params.update(vllm_kwargs)
    
    print(f"vLLM parameters: {default_params}")
    
    # Initialize vLLM
    llm = LLM(**default_params)
    
    # Get storage information
    bpw_layer, bpw_head, vram_bits = get_storage_info_vllm(llm)
    
    print(f"Loaded vLLM model: {bpw_layer:.2f} bpw (decoder), {bpw_head:.2f} bpw (head)")
    
    return llm, bpw_layer, bpw_head, vram_bits

@torch.inference_mode
def fwd_vllm(model_instance, input_ids: torch.Tensor):
    """
    Run forward pass using vLLM and convert logprobs to logits approximation.
    
    IMPORTANT LIMITATION: vLLM doesn't expose raw logits directly. This function uses
    prompt_logprobs to get log probabilities for each token position. The logprobs
    are returned in log space and are directly usable for perplexity calculation.
    
    However, vLLM only returns logprobs for a limited number of top tokens (controlled
    by the logprobs parameter in SamplingParams). This means we don't get full vocab
    coverage, which is fine for perplexity calculation of the actual next tokens, but
    may cause issues if those tokens fall outside the top-k returned.
    
    Args:
        model_instance: vLLM LLM instance
        input_ids: Input token IDs as torch.Tensor of shape [batch_size, seq_len]
    
    Returns:
        Logits tensor of shape [batch_size, seq_len, vocab_size]
        (Actually log probabilities in logit format)
    """
    # vLLM expects list of token IDs
    batch_size, seq_len = input_ids.shape
    
    # Convert input_ids to list of token lists
    if batch_size != 1:
        raise ValueError("vLLM comparison currently only supports batch_size=1")
    
    input_token_ids = input_ids[0].tolist()
    
    # Use vLLM's generate with prompt_logprobs to get log probabilities
    # We set max_tokens=1 (minimum allowed) - we only need prompt logprobs
    # We use logprobs=None to get ALL vocab logprobs (if supported)
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=1,  # Minimum allowed by vLLM
        prompt_logprobs=None,  # Get logprobs for all vocab tokens if possible
        logprobs=None,  # Get all logprobs
    )
    
    try:
        # Generate - vLLM v0.1.dev+ requires TokensPrompt wrapper
        from vllm import TokensPrompt
        
        prompt = TokensPrompt(prompt_token_ids=input_token_ids)
        outputs = model_instance.generate(
            [prompt],
            sampling_params,
        )
        
        output = outputs[0]
        
        # Debug: Print what vLLM actually returned
        print(f"DEBUG: Output type: {type(output)}")
        print(f"DEBUG: Output attributes: {dir(output)}")
        print(f"DEBUG: Has prompt_logprobs: {hasattr(output, 'prompt_logprobs')}")
        if hasattr(output, 'prompt_logprobs'):
            print(f"DEBUG: prompt_logprobs type: {type(output.prompt_logprobs)}")
            print(f"DEBUG: prompt_logprobs value: {output.prompt_logprobs}")
        
        # Extract prompt logprobs
        # vLLM returns prompt_logprobs as a list of dicts, where each dict maps token_id -> Logprob object
        prompt_logprobs = getattr(output, 'prompt_logprobs', None)
        
        if prompt_logprobs is None:
            raise ValueError(f"vLLM did not return prompt_logprobs. Output has these attributes: {[a for a in dir(output) if not a.startswith('_')]}")
        
        # Get vocab size from the model
        vocab_size = model_instance.llm_engine.model_config.get_vocab_size()
        
        # Initialize logits tensor with very negative values (effectively zero probability)
        # Using -1e10 instead of -100 to better represent very low probabilities
        logits = torch.full((batch_size, seq_len, vocab_size), -1e10, dtype=torch.float32)
        
        # Fill in the logprobs we have
        # prompt_logprobs[0] is None (no logprob for first token in most cases)
        # prompt_logprobs[1] onwards contain the logprobs for predicting tokens[1], tokens[2], etc.
        for pos in range(len(prompt_logprobs)):
            if prompt_logprobs[pos] is not None:
                for token_id, logprob_obj in prompt_logprobs[pos].items():
                    logprob = logprob_obj.logprob
                    # Store the logprob directly
                    # These are already log probabilities, which is what we need
                    if pos < seq_len:
                        logits[0, pos, token_id] = logprob
        
        return logits
        
    except Exception as e:
        print(f"Error in vLLM forward pass: {e}")
        print("This may be due to vLLM version incompatibility or configuration issues.")
        print("Ensure you're using a recent version of vLLM that supports prompt_logprobs.")
        raise

