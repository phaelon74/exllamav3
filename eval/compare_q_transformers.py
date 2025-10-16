import torch
import os
import json
from transformers import AutoTokenizer, AutoModelForCausalLM

try:
    from gptqmodel.nn_modules.qlinear.marlin import MarlinQuantLinear
    from gptqmodel.nn_modules.qlinear.tritonv2 import TritonV2QuantLinear
    from gptqmodel.nn_modules.qlinear.exllamav2 import ExllamaV2QuantLinear
except (ModuleNotFoundError, ImportError):
    MarlinQuantLinear = None
    TritonV2QuantLinear = None
    ExllamaV2QuantLinear = None

try:
    from aqlm import QuantizedLinear
except (ModuleNotFoundError, ImportError):
    QuantizedLinear = None

try:
    from awq.modules.linear import WQLinear_GEMM
except (ModuleNotFoundError, ImportError):
    WQLinear_GEMM = None

try:
    from vptq import VQuantLinear
except (ModuleNotFoundError, ImportError):
    VQuantLinear = None

try:
    from bitsandbytes.nn import Linear4bit
except (ModuleNotFoundError, ImportError):
    Linear4bit = None

try:
    from compressed_tensors.nn import CompressedLinear
except (ModuleNotFoundError, ImportError):
    CompressedLinear = None

def get_tensors_size(tensors):
    return 8 * sum(t.element_size() * t.numel() for t in tensors.values() if t is not None)

def get_tensor_size(tensor):
    return 8 * tensor.element_size() * tensor.numel()

def scan_gpu_tensors(obj, seen = None):
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    seen.add(obj_id)
    total_size = 0
    # If it's a GPU tensor, add its memory usage.
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

def get_storage_info(model):
    sum_bits = 0
    sum_numel = 0
    head_bpw = 0
    head_numel = 0
    
    # Filter out None values from optional quantization modules
    def check_isinstance(module, types_list):
        valid_types = [t for t in types_list if t is not None]
        if not valid_types:
            return False
        return isinstance(module, tuple(valid_types))
    
    compressed_count = 0
    linear_count = 0
    
    for name, module in model.named_modules():
        # Check for CompressedLinear by type or class name (fallback)
        is_compressed = (check_isinstance(module, [CompressedLinear]) or 
                        module.__class__.__name__ == 'CompressedLinear')
        
        if is_compressed:
            compressed_count += 1
            # Compressed-tensors quantization (NVFP4, etc.)
            # Sum all parameters in the compressed module
            module_bits = get_tensors_size(dict(module.named_parameters()))
            module_numel = module.in_features * module.out_features
            if module.out_features >= model.vocab_size * 0.9:
                head_bpw = module_bits / module_numel
                head_numel = module_numel
            else:
                sum_bits += module_bits
                sum_numel += module_numel
        elif check_isinstance(module, [Linear4bit]):
            if module.out_features >= model.vocab_size * 0.9:  # this is foolproof
                head_numel = module.in_features * module.out_features
                head_bpw = module.weight.numel() * 8
                head_bpw = (head_bpw + scan_gpu_tensors(module.quant_state) * 8) / head_numel
            else:
                sum_bits += module.weight.numel() * 8
                sum_bits += scan_gpu_tensors(module.quant_state) * 8
                sum_numel += module.in_features * module.out_features
        elif isinstance(module, torch.nn.Linear):
            # Skip CompressedLinear (handled above) - it inherits from Linear but has different attributes
            if module.__class__.__name__ == 'CompressedLinear':
                continue
            linear_count += 1
            # Regular linear layers
            if module.out_features >= model.vocab_size * 0.9:
                head_bpw = module.weight.element_size() * 8
                head_numel = module.weight.numel()
            else:
                sum_bits += get_tensor_size(module.weight)
                sum_numel +=  module.weight.numel()
        elif check_isinstance(module, [QuantizedLinear, VQuantLinear]):
            sum_bits += get_tensors_size(dict(module.named_parameters()))
            sum_numel += module.in_features * module.out_features
        elif check_isinstance(module, [WQLinear_GEMM]):
            sum_bits += get_tensors_size({
                "qweight": module.qweight,
                "qzeros": module.qzeros,
                "scales": module.scales,
            })
            sum_numel += module.in_features * module.out_features
        elif check_isinstance(module, [MarlinQuantLinear]):
            sum_bits += get_tensors_size({
                "g_idx": module.g_idx,
                "g_idx_sort_indices": module.g_idx_sort_indices,
                "qweight": module.qweight,
                "qzeros": module.qzeros,
                "scales": module.scales,
            })
            sum_numel += module.in_features * module.out_features
        elif check_isinstance(module, [TritonV2QuantLinear]):
            sum_bits += get_tensors_size({
                "g_idx": module.g_idx,
                "qweight": module.qweight,
                "qzeros": module.qzeros,
                "scales": module.scales,
            })
            sum_numel += module.in_features * module.out_features
        elif check_isinstance(module, [ExllamaV2QuantLinear]):
            sum_bits += get_tensors_size(module.q_tensors)
            sum_numel += module.in_features * module.out_features
    
    # Handle case where no quantized layers were found
    if sum_numel == 0:
        # Assume standard linear layers, estimate from model parameters
        print("Warning: Could not detect quantization format, estimating BPW from total parameters")
        return 16.0, 16.0, sum_bits  # Default to FP16
    
    print(f"DEBUG: Found {compressed_count} CompressedLinear layers, {linear_count} regular Linear layers")
    print(f"DEBUG: sum_bits={sum_bits}, sum_numel={sum_numel}, layer_bpw={sum_bits/sum_numel:.3f}")
    
    vram_bits = head_numel * head_bpw + sum_bits
    return sum_bits / sum_numel, head_bpw, vram_bits

@torch.inference_mode
def load_transformers(model_dir: str, auto = False, bf16 = False):
    # Check if this is a compressed-tensors model
    config_path = os.path.join(model_dir, "config.json")
    is_compressed = False
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
            if 'quantization_config' in config and config['quantization_config'].get('quant_method') == 'compressed-tensors':
                is_compressed = True
                print(f"DEBUG: Detected compressed-tensors model")
    
    load_kwargs = {
        "device_map": "auto" if auto else "cuda:0",
        "torch_dtype": torch.bfloat16 if bf16 else torch.half
    }
    
    # For compressed-tensors models, we need to trust remote code
    if is_compressed:
        load_kwargs["trust_remote_code"] = True
        print(f"DEBUG: Loading with trust_remote_code=True for compressed-tensors")
    
    model = AutoModelForCausalLM.from_pretrained(model_dir, **load_kwargs)
    
    # Verify the model loaded correctly
    print(f"DEBUG: Model type: {type(model)}")
    print(f"DEBUG: First layer type: {type(list(model.modules())[10])}")
    
    # Check what device the model is on
    if hasattr(model, 'device'):
        print(f"DEBUG: Model device: {model.device}")
    if hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
        print(f"DEBUG: Embedding device: {model.model.embed_tokens.weight.device}")
    
    # Check if using device_map="auto"
    if auto:
        print(f"DEBUG: Loaded with device_map='auto'")
        # Find first Linear/CompressedLinear layer device
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear) or module.__class__.__name__ == 'CompressedLinear':
                if hasattr(module, 'weight'):
                    print(f"DEBUG: First linear layer ({name}) is on device: {module.weight.device}")
                break
    
    bpw_layer, bpw_head, vram_bits = get_storage_info(model)
    return model, bpw_layer, bpw_head, vram_bits

@torch.inference_mode
def load_transformers_auto(model_dir: str):
    return load_transformers(model_dir, auto = True)

@torch.inference_mode
def load_transformers_auto_bf16(model_dir: str):
    return load_transformers(model_dir, auto = True, bf16 = True)

@torch.inference_mode
def fwd_transformers(model_instance, input_ids: torch.Tensor):
    input_ids = input_ids.to("cuda:0")
    
    # Check if model has compressed layers
    has_compressed = False
    for name, module in model_instance.named_modules():
        if module.__class__.__name__ == 'CompressedLinear':
            has_compressed = True
            # Check if the module has a forward method
            if hasattr(module, 'forward'):
                print(f"DEBUG: CompressedLinear found: {name}, forward method: {type(module.forward)}")
            if hasattr(module, 'weight'):
                print(f"DEBUG: CompressedLinear has weight attribute, dtype: {module.weight.dtype}, shape: {module.weight.shape}")
            else:
                print(f"DEBUG: CompressedLinear does NOT have weight attribute")
                if hasattr(module, 'packed_weight'):
                    print(f"DEBUG: CompressedLinear has packed_weight, dtype: {module.packed_weight.dtype}")
            break  # Just check the first one
    
    if has_compressed:
        print(f"DEBUG: Model has compressed layers, checking if dequantization is happening...")
    
    output = model_instance(input_ids)
    
    # Debug: Check if logits look reasonable
    logits = output.logits
    print(f"DEBUG FWD: logits shape={logits.shape}, min={logits.min():.3f}, max={logits.max():.3f}, mean={logits.mean():.3f}")
    print(f"DEBUG FWD: First 10 logits: {logits[0, 0, :10].tolist()}")
    
    return logits

@torch.inference_mode
def tokenize_transformers(tokenizer_dir: str, text: str):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    output = tokenizer(text, return_tensors="pt")
    return output.input_ids
