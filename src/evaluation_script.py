from gensim.models import CoherenceModel
from gensim.corpora import Dictionary
from sklearn.metrics import silhouette_score
import numpy as np
import os
import pandas as pd
from .scripts.preprocessing import load_and_preprocess_data, tokenize_reviews, vectorize_sequences
from .scripts.topic_modelling import apply_bertopic, return_topic_sequences_csv, visualize, print_model_info
def main():

    params = {
        'new_reviews': 0,  # 0 for old reviews, 1 new reviews
        'product_review_count': 10, #596,
        'tokens': 1,
        # delete?
        'nan': True, 
        'emojis': True,
        'contractions': True,
        'special_chars': True,
        'whitespaces': True,
        'stopwords': True,
        'lemmatization': True,
        'lowercase': True,
        'emails_and_urls': True,
        'nouns': False,
        'adj': False,
        'numbers': True,
        'most_frequent': 0
    }

    base_dir = os.path.dirname(os.path.abspath(__file__))
    processed_dir = os.path.join(base_dir, '../data/processed')

    reviews_cleaned = pd.read_csv(os.path.join(processed_dir, f'size_{params["product_review_count"]}_processed.csv'))['reviewText']

    tokens = params['tokens']
    # 0 to tokenize in sentences
    # n>0 to tokenize in n-grams
    print("-------------------------")
    # tokenizetion of reviews
    sequences_list, sequences_series = tokenize_reviews(reviews_cleaned, tokens, params['stopwords'], params['lemmatization']) 
    # sequences_list is a python list
    # sequences_series is a pandas series

    ######################BERTOPIC############################
    model = "all-MiniLM-L6-v2"
    reduced_topics = 10
    # application of BERTopic  modeling
    topic_model, umap_model, embeddings, topics = apply_bertopic(sequences_list, model, reduced_topics)
    topic_info = topic_model.get_topic_info()
    print(topic_info)
    
    # Coherence Model
    topics_cm = topic_model.get_topics()
    topic_words = [[word for word, _ in topic] for _, topic in topics_cm.items() if topic]
    sequences = [sequence.split() for sequence in sequences_list]
    dictionary = Dictionary(sequences)
    coherence_model = CoherenceModel(topics=topic_words,
                                     texts=sequences,
                                     dictionary=dictionary,
                                     coherence="c_v")
    print(f"Coherencia: {coherence_model.get_coherence()}")

    # Silhouette Score
    umap_embeddings = topic_model.umap_model.transform(embeddings)
    indices = [index for index, topic in enumerate(topics) if topic != -1]
    X = umap_embeddings[np.array(indices)]
    labels = [topic for index, topic in enumerate(topics) if topic != -1]
    print(f"Silhouette Score: {silhouette_score(X, labels)}")

if __name__ == "__main__":
    main()

# Bertopic para distintos parametros
# subhipotesis
# acotando el calculo

# antes:
# lo mismo para lda

# y comprar ambos

# Coherence score 
# Siluete score
