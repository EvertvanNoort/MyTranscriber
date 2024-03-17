import spacy
from gensim.models import HdpModel
from gensim.corpora import Dictionary
from gensim.parsing.preprocessing import preprocess_string

# Load the spaCy model
# nlp = spacy.load("nl_core_news_sm")  # For Dutch
# nlp = spacy.load("nl_core_news_lg")  # For Dutch
nlp = spacy.load("en_core_web_sm")  # For English

def get_important_sentences(file_path, summary_path, prob_threshold=0.8):
# def get_important_sentences(text, prob_threshold=0.65):

    text = read_text_from_file(file_path)
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents]
    processed_sentences = [preprocess_string(sent) for sent in sentences]

    dictionary = Dictionary(processed_sentences)
    corpus = [dictionary.doc2bow(sent) for sent in processed_sentences]

    hdp = HdpModel(corpus, id2word=dictionary)
    topic_distributions = [hdp[bow] for bow in corpus]

    important_sentences_ordered = []
    seen_indices = set()
    for i, distribution in enumerate(topic_distributions):
        for _, prob in distribution:
            if prob > prob_threshold and i not in seen_indices:
                important_sentences_ordered.append(sentences[i])
                seen_indices.add(i)
                # print(important_sentences_ordered)

    important_sentences_combined = " ".join(important_sentences_ordered)
    print("Extractive summary written to:",summary_path)

    with open(summary_path, 'w', encoding='utf-8') as file:
        file.write(important_sentences_combined)

def read_text_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()