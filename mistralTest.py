import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import AutoTokenizer, MistralForCausalLM
device = "cuda" # the device to load the model onto

# model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1") #, torch_dtype=torch.float16, attn_implementation="flash_attention_2")
# tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")


model = MistralForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1", torch_dtype=torch.float16, attn_implementation="flash_attention_2")
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

prompt = "Hey, are you conscious? Can you talk to me?"
inputs = tokenizer(prompt, return_tensors="pt")

# Generate
generate_ids = model.generate(inputs.input_ids, max_length=30)
tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
