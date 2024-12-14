import os
import pandas as pd

from .scripts.summary_gen import summary_gen
from .scripts.preprocessing import load_and_preprocess_data, tokenize_reviews, vectorize_sequences
from .scripts.topic_modelling import apply_bertopic, return_topic_sequences_csv, visualize, print_model_info
from .scripts.sentiment_analysis import sentiment_analysis
from .scripts.config import params

def main():

    base_dir = os.path.dirname(os.path.abspath(__file__))
    processed_dir = os.path.join(base_dir, '../data/processed')

    reviews_cleaned = pd.read_csv(f"../data/processed/'size_{params['product_review_count']}_processed.csv")['reviewText']

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
    topic_model = apply_bertopic(sequences_list, model, reduced_topics)
    
    print_model_info(topic_model, sequences_list, model)

    # visualize(topic_model, sequences_list, model)

    return_topic_sequences_csv(topic_model, sequences_series)

    ######################TEXTBLOB############################
    source = pd.read_csv(os.path.join(processed_dir,'sequences_topics.csv'))
    # sentiment_analysis(source)

    ######################SUMMARY GENERATOR############################


if __name__ == "__main__":
    main()