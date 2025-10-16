import torch
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
    
    for name, module in model.named_modules():
        if check_isinstance(module, [CompressedLinear]):
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
    
    vram_bits = head_numel * head_bpw + sum_bits
    return sum_bits / sum_numel, head_bpw, vram_bits

@torch.inference_mode
def load_transformers(model_dir: str, auto = False, bf16 = False):
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        device_map = "auto" if auto else "cuda:0",
        torch_dtype = torch.bfloat16 if bf16 else torch.half
    )
    bpw_layer, bpw_head, vram_bits = get_storage_info(model)
    return model, bpw_layer, bpw_head, vram_bits

@torch.inference_mode
def load_transformers_auto(model_dir: str):
    return load_transformers(model_dir, auto = True)

@torch.inference_mode
def load_transformers_auto_bf16(model_dir: str):
    return load_transformers(model_dir, auto = True, bf16 = True)

@torch.inference_mode
def load_transformers_tensor_parallel(model_dir: str):
    """
    Load model with tensor parallelism using device_map="auto".
    Supports compressed-tensors quantization (NVFP4, etc.)
    Args can be passed as [model_dir, kwargs_dict] for advanced configs.
    """
    # Support both string and [string, dict] formats
    if isinstance(model_dir, list):
        actual_model_dir = model_dir[0]
        kwargs = model_dir[1] if len(model_dir) > 1 else {}
    else:
        actual_model_dir = model_dir
        kwargs = {}
    
    # Default parameters for tensor parallel loading
    default_params = {
        "device_map": "auto",
        "dtype": torch.bfloat16,  # Use 'dtype' not 'torch_dtype' (deprecated)
        "trust_remote_code": True,  # Often needed for compressed-tensors models
        "low_cpu_mem_usage": True,
    }
    
    # Override defaults with user-provided kwargs
    # Handle dtype/torch_dtype string conversion
    for dtype_key in ["dtype", "torch_dtype"]:
        if dtype_key in kwargs and isinstance(kwargs[dtype_key], str):
            dtype_str = kwargs[dtype_key]
            dtype_value = None
            if dtype_str == "bfloat16":
                dtype_value = torch.bfloat16
            elif dtype_str == "float16":
                dtype_value = torch.float16
            elif dtype_str == "float32":
                dtype_value = torch.float32
            elif dtype_str == "auto":
                dtype_value = "auto"
            
            if dtype_value is not None:
                # Use 'dtype' as the preferred parameter name
                kwargs.pop(dtype_key, None)
                kwargs["dtype"] = dtype_value
    
    default_params.update(kwargs)
    
    print(f"Loading model with tensor parallelism: {actual_model_dir}")
    print(f"Parameters: {default_params}")
    
    model = AutoModelForCausalLM.from_pretrained(
        actual_model_dir,
        **default_params
    )
    
    bpw_layer, bpw_head, vram_bits = get_storage_info(model)
    return model, bpw_layer, bpw_head, vram_bits

@torch.inference_mode
def fwd_transformers(model_instance, input_ids: torch.Tensor):
    input_ids = input_ids.to("cuda:0")
    output = model_instance(input_ids)
    return output.logits

@torch.inference_mode
def fwd_transformers_auto(model_instance, input_ids: torch.Tensor):
    """
    Forward pass for models loaded with device_map="auto".
    Lets the model handle device placement automatically.
    """
    # Don't force device placement - let the model handle it
    # The model already knows where its tensors are
    if hasattr(model_instance, 'device'):
        # If model has a .device attribute, use it
        input_ids = input_ids.to(model_instance.device)
    elif hasattr(model_instance, 'model') and hasattr(model_instance.model, 'embed_tokens'):
        # Otherwise, use the device of the embedding layer
        input_ids = input_ids.to(model_instance.model.embed_tokens.weight.device)
    else:
        # Fallback: assume first device
        input_ids = input_ids.to('cuda:0')
    
    output = model_instance(input_ids)
    return output.logits

@torch.inference_mode
def tokenize_transformers(tokenizer_dir: str, text: str):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    output = tokenizer(text, return_tensors="pt")
    return output.input_ids
