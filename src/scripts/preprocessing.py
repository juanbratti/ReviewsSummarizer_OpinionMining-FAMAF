import os
import pandas as pd
from ..utils import clean_reviews, get_product_with_n_reviews, split_into_sentences, tokenize_reviews_to_sequences, lemmatisation_stopwords_series
from sklearn.feature_extraction.text import CountVectorizer
from .config import params

####################### DATA PREPROCESS #######################

def load_and_preprocess_data(params):
    """
    Loads raw file and preprocesses the data. It:
    - extracts the product with a specific number of reviews
    - cleans the reviews (removes emojis, contractions, special characters and extra whitespaces, stopwords, nounds, adj and lemmatizes)
    - saves the processed data to a new file

    Args:
        params (dict): dictionary containing the file path, number of reviews, and cleaning options

    Returns:
        dataset (pd.DataFrame): dataset in the filepath as DataFrame
        reviews_cleaned (pd.Series): Cleaned reviews for the product with the specified number of reviews
        product_id (str): Product ID
    """
    new_reviews = params['new_reviews']

    base_dir = os.path.dirname(os.path.abspath(__file__))

    if not new_reviews:
        file_path = os.path.join(base_dir, '../../data/parsed_input_file.csv')        
        product_review_count = params['product_review_count']
    else:
        file_path = os.path.join(base_dir, '../../data/parsed_input_file2.csv')        
    
    # load of the dataset
    dataset = pd.read_csv(file_path)
    
    if not new_reviews:
        # retrieve the product with a specific number of reviews
        product_id = get_product_with_n_reviews(file_path, product_review_count)

        # extraction and cleaning the reviews
        reviews_raw = dataset[dataset['productId'] == product_id]['reviewText']

    else:
        # first column only
        reviews_raw = dataset.iloc[:, 0]

    params_clean_review = {
        'nan': params['nan'],
        'emojis': params['emojis'],
        'contractions': params['contractions'],
        'special_chars': params['special_chars'],
        'whitespaces': params['whitespaces'],
        'stopwords': params['stopwords'],
        'lemmatization': params['lemmatization'],
        'lowercase': params['lowercase'],
        'emails_and_urls': params['emails_and_urls'],
        'nouns': params['nouns'],
        'adj': params['adj'],
        'numbers': params['numbers'],
        'most_frequent': params['most_frequent'],
        'tokens':params['tokens']
    }
    
    reviews_cleaned = clean_reviews(reviews_raw, params_clean_review)

    reviews_cleaned = reviews_cleaned[reviews_cleaned.str.strip() != '']
    
    processed_dir = os.path.join(base_dir, '../../data/processed')

    # save the file to a csv
    if not new_reviews:
        reviews_cleaned.to_csv(os.path.join(processed_dir, f'size_{params["product_review_count"]}_processed.csv'), index=False)
    else: 
        reviews_cleaned.to_csv(os.path.join(processed_dir, 'new_reviews'), index=False)

    return dataset, reviews_cleaned

def tokenize_reviews(reviews_cleaned, tokens, stopwords, lemmatization):
    """
    Tokenizes the cleaned reviews and returns a list of sequences.

    Args:
        reviews_cleaned (pd.Series): cleaned reviews (preprocessed reviews)
        tokens (int): 0 to tokenize in sentences, n>0 to tokenize in n-grams
        
    Returns:
        sequences_list (list): python list of tokenized sequences
    """

    # tokenization of the clean reviews
    if tokens == 0:
        reviews_tokens = split_into_sentences(reviews_cleaned)
    else:
        reviews_tokens = tokenize_reviews_to_sequences(reviews_cleaned, tokens)

    # remove stopwords and lemmatize
    reviews_tokens = lemmatisation_stopwords_series(reviews_tokens, stopwords, lemmatization)

    # remove empty rows
    # reviews_tokens = reviews_tokens[reviews_tokens != '']

    # convertion of the tokenized series to a list of strings
    sequences_list = reviews_tokens.tolist()

    return sequences_list, reviews_tokens


def vectorize_sequences(sequences_list):
    """
    Vectorizes the sequences using CountVectorizer.

    Args:
        sequences_list (list): list of tokenized sequences

    Returns:
        vectorizer (CountVectorizer): trained CountVectorizer model
        dtm: document-term matrix
    """


    # vectorization
    vectorizer = CountVectorizer()
    dtm = vectorizer.fit_transform(sequences_list)

    return vectorizer, dtm

####################### MAIN #######################    
def main():
    load_and_preprocess_data(params)

if __name__ == "__main__":
    main()