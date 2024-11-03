import pandas as pd
from models.guidedlda import apply_guidedlda
from scripts.preprocessing import load_and_preprocess_data, tokenize_reviews, vectorize_sequences
from scripts.topic_modelling import apply_bertopic, visualize, print_model_info
from models.topics_kmeans import *
from models.topics_lda import *

def main():

    params = {
        'new_reviews': 1,  # 0 for old reviews, 1 new reviews
        'product_review_count': 40,
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

    raw_dataset, reviews_cleaned = load_and_preprocess_data(params)

    tokens = 5
    # 0 to tokenize in sentences
    # n>0 to tokenize in n-grams

    # tokenizetion of reviews
    sequences_list, sequences_series = tokenize_reviews(reviews_cleaned, tokens, params['stopwords'], params['lemmatization']) 
    # sequences_list is a python list
    # sequences_series is a pandas series
    seed_topic_list = [
        ["price", "cheap", "expensive", "value", "affordable", "cost-effective", "budget-friendly", "inexpensive", "pricy", "overpriced", "costly"],
        ["quality", "good", "bad", "durable", "high-quality", "low-quality", "well-made", "fragile", "sturdy", "weak", "reliable", "unreliable"],
        ["use", "easy", "difficult", "works", "intuitive", "counterintuitive", "straightfoward", "complicated", "efficient", "inefficient", "unreliable"],
        ["design", "nice", "ugly", "aesthetic", "stylish", "unstylish", "attractive", "modern", "outdated", "elegant", "tasteful", "tasteless"]
    ]
    # Select model
    ######################BERTOPIC############################
    model = "all-MiniLM-L6-v2"
    #model = "umap"
    reduced_topics = 10
    # application of BERTopic  modeling
    topic_model = apply_bertopic(sequences_list, seed_topic_list, model, reduced_topics)

    print_model_info(topic_model, sequences_list, model)

    visualize(topic_model, sequences_list, model)

    # ###################GUIDEDLDA##############################
    # topic_model_guidedlda = apply_guidedlda(sequences_list, seed_topic_list)


    # ####################KMEANS################################
    # n_clusters = 5
    # important_words = 5
    # topic_model, topics = apply_kmeans(sequences_list, n_clusters)
    
    # print_model_info_kmeans(topic_model, sequences_list, n_clusters, important_words)

    # visualize_kmeans(topic_model, sequences_list)


    # ####################LDA###################################
    # num_topics = 10
    # topic_model = apply_lda(sequences_list, num_topics)
    # shown_docs = 20
    # print_model_info_lda(topic_model, sequences_list, shown_docs)


if __name__ == "__main__":
    main()