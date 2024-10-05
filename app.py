"""
This module performs X tasks or defines Y classes/functions.

Details about the module can go here.
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login


login("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")

model = "meta-llama/Llama-3.2-1B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model)
model = AutoModelForCausalLM.from_pretrained(model)
prompt = "write a code to add two numbers"
inputs = tokenizer(prompt, return_tensors="pt")

output = model.generate(
    inputs.input_ids, 
    max_new_tokens=1000,
    do_sample=True,
    top_k=50,
    top_p=0.95,
    temperature=0.9
    )

generated_text = tokenizer.decode(output[0])
print(generated_text)
print("done")