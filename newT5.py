from transformers import T5Tokenizer, T5ForConditionalGeneration

# Load pre-trained T5 model and tokenizer
model_name = "t5-large"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)


# Input text to be summarized
# Function to read text from a file
def read_text_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()
file_path = '/home/evert/Desktop/audio/1_transcript.txt'
input_text = read_text_from_file(file_path)

# Tokenize and summarize the input text using T5
inputs = tokenizer.encode("summarize: " + input_text, return_tensors="pt", max_length=1024, truncation=True)
summary_ids = model.generate(inputs, max_length=512, min_length=150, length_penalty=2.0, num_beams=4, early_stopping=True)

# Decode and output the summary
summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
print("Original Text:")
print(input_text)
print("\nSummary:")
print(summary)