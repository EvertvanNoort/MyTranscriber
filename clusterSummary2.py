from gensim.corpora.dictionary import Dictionary
from gensim.models.ldamodel import LdaModel
from gensim.parsing.preprocessing import preprocess_string
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from transformers import pipeline
import spacy
import torch
import numpy as np

device = 0 if torch.cuda.is_available() else -1

# Load the spaCy transformer model
# nlp = spacy.load('en_core_web_trf')
nlp = spacy.load('en_core_web_sm')  # or 'nl_core_news_sm' for Dutch

# Initialize the summarization pipeline
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", device=device)
# summarizer = pipeline("summarization", model="t5-3b", device=device)

# Function to divide text into paragraphs using LDA
def get_complete_topic_distribution(lda_model, bow, num_topics):
    topic_distribution = dict(lda_model.get_document_topics(bow, minimum_probability=0))
    return [topic_distribution.get(i, 0) for i in range(num_topics)]

def divide_into_lda_based_paragraphs(text, num_topics=5, threshold=0.05):
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents]
    processed_sentences = [preprocess_string(sent) for sent in sentences]

    # Create a dictionary representation of the sentences
    dictionary = Dictionary(processed_sentences)
    corpus = [dictionary.doc2bow(sent) for sent in processed_sentences]

    # LDA model
    lda = LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=10)

    paragraphs = []
    current_paragraph = [sentences[0]]

    for i in range(1, len(sentences)):
        bow = dictionary.doc2bow(preprocess_string(sentences[i]))
        topic_distribution_i = get_complete_topic_distribution(lda, bow, num_topics)
        
        bow_prev = dictionary.doc2bow(preprocess_string(sentences[i-1]))
        topic_distribution_prev = get_complete_topic_distribution(lda, bow_prev, num_topics)

        # Compute similarity in topic distribution
        similarity = sum([min(topic_distribution_i[k], topic_distribution_prev[k]) 
                          for k in range(num_topics)])
        
        if similarity < threshold:
            paragraphs.append(" ".join(current_paragraph))
            current_paragraph = [sentences[i]]
        else:
            current_paragraph.append(sentences[i])

    if current_paragraph:
        paragraphs.append(" ".join(current_paragraph))

    return paragraphs

from gensim.models import HdpModel

# Function to create an HDP model
def create_hdp_model(dictionary, corpus):
    hdp = HdpModel(corpus, id2word=dictionary)
    return hdp

# Function to divide text into paragraphs using HDP-based topics
def divide_into_hdp_based_paragraphs(text, threshold=0.15):
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

# Usage in your main program

# Function to read text from a file
def read_text_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

# Read text from file
file_path = '/home/evert/Desktop/wiki.txt'
text = read_text_from_file(file_path)

# Choose your segmentation method here
# segments = divide_into_lda_based_paragraphs(text)
segments = divide_into_hdp_based_paragraphs(text)

# Function to count the number of words in the text
def count_words_in_text(text):
    words = text.split()
    return len(words)

# Summarize each segment and combine
full_summary = []
for segment in segments:
    word_count = count_words_in_text(segment)
    segment_summary = summarizer(segment, max_length= word_count, min_length=round(word_count/10), length_penalty=2.0, num_beams=4, early_stopping=True)
    full_summary.append("Original text: \n")
    full_summary.append(segment)
    full_summary.append("\n \n")
    full_summary.append("Summary: \n")
    full_summary.append(segment_summary[0]['summary_text'])
    full_summary.append("\n \n")

# Combine all segment summaries into one
coherent_summary = " ".join(full_summary)

print(coherent_summary)

# Write the coherent summary to a file
output_file_path = '/home/evert/Desktop/coherent_summary.txt'

with open(output_file_path, 'w', encoding='utf-8') as file:
    file.write(coherent_summary)

print(f"Coherent summary written to {output_file_path}")
