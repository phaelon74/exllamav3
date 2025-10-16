#!/usr/bin/env python3
"""Quick test to see if compressed-tensors model produces reasonable output"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "/media/fmodels/TheHouseOfTheDude/Llama-3.1-8B-Instruct/W4A16/"

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
)

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_path)

print("\nTesting generation...")
prompt = "The capital of France is"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")

print(f"Prompt: {prompt}")
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=20, do_sample=False)
    
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"Generated: {generated_text}")

# Also check logits for first token
with torch.no_grad():
    logits = model(**inputs).logits
    next_token_logits = logits[0, -1, :]
    top5_tokens = torch.topk(next_token_logits, 5)
    print(f"\nTop 5 next token predictions:")
    for score, token_id in zip(top5_tokens.values, top5_tokens.indices):
        token_text = tokenizer.decode([token_id])
        print(f"  {token_text!r}: {score:.3f}")

