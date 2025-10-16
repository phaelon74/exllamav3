#!/usr/bin/env python3
"""Quick script to check vLLM version and API compatibility"""

try:
    import vllm
    print(f"vLLM version: {vllm.__version__}")
    
    # Check if TokensPrompt is available
    try:
        from vllm import TokensPrompt
        print("✓ TokensPrompt available (newer API)")
    except ImportError:
        print("✗ TokensPrompt not available (older API)")
    
    # Check LLM.generate signature
    from vllm import LLM
    import inspect
    sig = inspect.signature(LLM.generate)
    print(f"\nLLM.generate() parameters: {list(sig.parameters.keys())}")
    
except ImportError as e:
    print(f"vLLM not installed: {e}")

