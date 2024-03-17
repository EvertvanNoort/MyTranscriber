import accelerate
import bitsandbytes
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import AutoTokenizer, MistralForCausalLM
from transformers import BitsAndBytesConfig

# Check if GPU is available and set device
device = 0 if torch.cuda.is_available() else -1

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# Function to read text from a file
def read_text_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

# # Read text from file
file_path = '/home/evert/Desktop/QuestionAnswering/dialoognl.txt'
text = read_text_from_file(file_path)

checkpoint = "mistralai/Mistral-7B-v0.1"

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint, quantization_config=bnb_config)

# model = MistralForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1", torch_dtype=torch.float16, attn_implementation="flash_attention_2")
# tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

prompt = "A patient went to the doctor and got a medicine, he felt better right after, but fell asleep on the bus and missed his stop."
inputs = tokenizer(prompt, return_tensors="pt").to(device)

# Generate
generate_ids = model.generate(inputs.input_ids, max_length=50).to(device)
print(tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0])

# print(answer(outputs[0]))