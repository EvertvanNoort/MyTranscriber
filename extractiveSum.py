from gensim.summarization import summarize
from gensim.models import HdpModel
from gensim.corpora.dictionary import Dictionary
from gensim.parsing.preprocessing import preprocess_string
import spacy
import numpy as np

# Load the spaCy model
nlp = spacy.load('nl_core_news_lg')  # or 'nl_core_news_sm' for Dutch

# Function to create an HDP model
def create_hdp_model(dictionary, corpus):
    hdp = HdpModel(corpus, id2word=dictionary)
    return hdp

# Function to divide text into paragraphs using HDP-based topics
def divide_into_hdp_based_paragraphs(text, threshold=0.05):
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents]
    processed_sentences = [preprocess_string(sent) for sent in sentences]

    # Create a dictionary and corpus
    dictionary = Dictionary(processed_sentences)
    corpus = [dictionary.doc2bow(sent) for sent in processed_sentences]

    # Create HDP model
    hdp = create_hdp_model(dictionary, corpus)

    # Get a reasonable number of topics
    hdp_topics = hdp.print_topics(num_topics=-1, num_words=5)
    num_topics = len(hdp_topics)

    paragraphs = []
    current_paragraph = [sentences[0]]

    for i in range(1, len(sentences)):
        bow = dictionary.doc2bow(preprocess_string(sentences[i]))
        topic_distribution_i = [val for _, val in hdp[bow]]

        bow_prev = dictionary.doc2bow(preprocess_string(sentences[i-1]))
        topic_distribution_prev = [val for _, val in hdp[bow_prev]]

        # Ensure same length topic distributions
        len_diff = len(topic_distribution_i) - len(topic_distribution_prev)
        if len_diff > 0:
            topic_distribution_prev.extend([0]*len_diff)
        elif len_diff < 0:
            topic_distribution_i.extend([0]*abs(len_diff))

        # Compute similarity in topic distribution
        similarity = sum([min(topic_distribution_i[k], topic_distribution_prev[k]) 
                          for k in range(max(len(topic_distribution_i), len(topic_distribution_prev)))])
        
        if similarity < threshold:
            paragraphs.append(" ".join(current_paragraph))
            current_paragraph = [sentences[i]]
        else:
            current_paragraph.append(sentences[i])

    if current_paragraph:
        paragraphs.append(" ".join(current_paragraph))

    return paragraphs

# Function to read text from a file
def read_text_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

# Function to count the number of words in the text
def count_words_in_text(text):
    words = text.split()
    return len(words)

# Read text from file
file_path = '/home/evert/Desktop/MachineLearningNL.txt'
text = read_text_from_file(file_path)

# Segment the text
segments = divide_into_hdp_based_paragraphs(text)

# Summarize each segment using Gensim's extractive summarization and combine
full_summary = []
for segment in segments:
    try:
        # Using Gensim's summarize function on each segment
        summary_text = summarize(segment, ratio=0.05)  # Adjust the ratio as needed
        if summary_text:
            full_summary.append(summary_text)
            full_summary.append("\n \n")
    except ValueError as e:
        print(f"Error during summarization: {e}")

# Combine all segment summaries into one
coherent_summary = " ".join(full_summary)

print(coherent_summary)

# Write the coherent summary to a file
output_file_path = '/home/evert/Desktop/coherent_summary.txt'
with open(output_file_path, 'w', encoding='utf-8') as file:
    file.write(coherent_summary)

print(f"Coherent summary written to {output_file_path}")
