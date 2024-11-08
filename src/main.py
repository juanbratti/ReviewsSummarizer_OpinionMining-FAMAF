import pandas as pd
from scripts.preprocessing import load_and_preprocess_data, tokenize_reviews, vectorize_sequences
from scripts.topic_modelling import apply_bertopic, visualize, print_model_info

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

    tokens = 0
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
    topic_model = apply_bertopic(sequences_list, model, reduced_topics)
    

    print_model_info(topic_model, sequences_list, model)

    visualize(topic_model, sequences_list, model)


if __name__ == "__main__":
    main()