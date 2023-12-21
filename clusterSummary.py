from gensim.models import HdpModel
from gensim.corpora.dictionary import Dictionary
from gensim.parsing.preprocessing import preprocess_string
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from transformers import pipeline
import spacy
import torch
import numpy as np

device = 0 if torch.cuda.is_available() else -1
# device = -1

# Load the spaCy transformer model
# nlp = spacy.load('en_core_web_trf')
# nlp = spacy.load('en_core_web_sm')  # or 'nl_core_news_sm' for Dutch
# nlp = spacy.load('nl_core_news_sm')  # or 'nl_core_news_sm' for Dutch
nlp = spacy.load('nl_core_news_lg')  # or 'nl_core_news_sm' for Dutch

# Initialize the summarization pipeline
# summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=device)
# summarizer = pipeline("summarization", model="google/pegasus-cnn_dailymail", device=device)
# summarizer = pipeline("summarization", model="t5-base", device=device)
summarizer = pipeline("summarization", model="t5-large", device=device)
# summarizer = pipeline("summarization", model="t5-3b", device=device)
# summarizer = pipeline("summarization", model="yhavinga/t5-v1.1-base-dutch-cnn-test", device=device)

# Function to create an HDP model
def create_hdp_model(dictionary, corpus):
    hdp = HdpModel(corpus, id2word=dictionary)
    return hdp

# Function to divide text into paragraphs using HDP-based topics
def divide_into_hdp_based_paragraphs(text, threshold=0.035):
# def divide_into_hdp_based_paragraphs(text, threshold=0.01):
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents]
    processed_sentences = [preprocess_string(sent) for sent in sentences]

    # Create a dictionary and corpus
    dictionary = Dictionary(processed_sentences)
    corpus = [dictionary.doc2bow(sent) for sent in processed_sentences]

    # Create HDP model
    hdp = create_hdp_model(dictionary, corpus)

    # Get a reasonable number of topics
    hdp_topics = hdp.print_topics(num_topics=-1)#, num_words=5)
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
        paragraphs.append("\n")
    return paragraphs

# Function to read text from a file
def read_text_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def discard_incomplete_sentences(text):
    doc = nlp(text)
    sentences = list(doc.sents)
    processed_text = ' '.join([sentence.text for sentence in sentences if sentence.text.strip().endswith((',','.', '!', '?'))])

    return processed_text

# Function to count the number of words in the text
def count_words_in_text(text):
    words = text.split()
    return len(words)

# Read text from file
file_path = '/home/evert/Desktop/MachineLearningNL.txt'
text = read_text_from_file(file_path)

# Choose your segmentation method here
segments = divide_into_hdp_based_paragraphs(text)

# Summarize each segment and combine
full_summary = []
for segment in segments:
    word_count = count_words_in_text(segment)

    # Check if the segment is too short for summarization
    if word_count > 25:  # Example threshold, adjust as needed
        try:
            # Ensure max_length is greater than the segment length
            min_length = max(10, round(word_count/3))
            segment_summary = summarizer(segment, max_length=word_count, min_length=min_length, length_penalty= 1.0, num_beams=10, early_stopping=False)
            # segment_summary = summarizer(segment, max_length=word_count, min_length=min_length, num_beams=1)
            summary_text = segment_summary[0]['summary_text']
            
            summary_text = discard_incomplete_sentences(summary_text)

            full_summary.append(summary_text)
            full_summary.append("\n")
        except Exception as e:
            print(f"Error during summarization: {e}")
    else:
        full_summary.append(segment)

# Combine all segment summaries into one
coherent_summary = " ".join(full_summary)
structured_text = " ".join(segments)

# Write the coherent summary to a file
output_file_path = '/home/evert/Desktop/coherent_summary.txt'
output_seg_path = '/home/evert/Desktop/segments.txt'

with open(output_file_path, 'w', encoding='utf-8') as file:
    file.write(coherent_summary)

print(f"Coherent summary written to {output_file_path}")

with open(output_seg_path, 'w', encoding='utf-8') as file:
    file.write(structured_text)

print(f"Segments written to {output_seg_path}")
