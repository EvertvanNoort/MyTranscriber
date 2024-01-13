import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from collections import defaultdict

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Function to read text from a file
def read_text_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

# Function to score sentences based on word frequencies
def score_sentences(sentences, freq_table):
    sentence_scores = defaultdict(int)
    for sentence in sentences:
        word_count_in_sentence = len(word_tokenize(sentence))
        for word, freq in freq_table.items():
            if word in sentence.lower():
                sentence_scores[sentence] += freq
    return sentence_scores

# Function to summarize text using NLTK
def nltk_summarize(text, max_sentences=20):
    sentences = sent_tokenize(text)

    # Calculate word frequencies
    freq_table = defaultdict(int)
    words = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words and word.isalpha()]

    for word in words:
        freq_table[word] += 1

    # Score sentences
    sentence_scores = score_sentences(sentences, freq_table)

    # Sort sentences by score and pick the top ones
    summary_sentences = sorted(sentence_scores, key=sentence_scores.get, reverse=True)[:max_sentences]
    return ' '.join(summary_sentences)

# Read text from file
# file_path = '/path/to/your/file.txt'  # Replace with your file path
file_path = '/home/evert/Desktop/audio/1_transcript.txt'
text = read_text_from_file(file_path)

# Summarize the text
summary = nltk_summarize(text)

# Print the summary
print(summary)

# Optionally, write the summary to a file
# output_file_path = '/path/to/your/summary.txt'  # Replace with your desired output path
output_file_path = '/home/evert/Desktop/coherent_summary.txt'
with open(output_file_path, 'w', encoding='utf-8') as file:
    file.write(summary)

print(f"Summary written to {output_file_path}")
