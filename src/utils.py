import csv # this is for the creation of the resulting CSV file
import pandas as pd
import emoji
import spacy
import contractions


# --------------------------------------------- FUNCTIONS

def get_reviews_per_product(input_file):
    """
    From a parsed csv file, returns the amount of reviews per productId (in a Series).
    
    Args:
        input_file (str): The path to the input text file.

    Returns: 
        reviews_per_product (pd.Series): The amount of reviews per productId.
    """

    # load the csv into a pandas df
    df = pd.read_csv(input_file)

    reviews_per_product = df.groupby('productId').size()
    return reviews_per_product

# Now, we want the product with the max amount of reviews

def get_product_with_max_reviews(input_file):

    """
    From a parsed csv file, returns the productId that has the most reviews.
    
    Args:
        input_file (str): The path to the input text file.

    Returns: 
        product_id (str): The productId with the most reviews.
        max_reviews (int): The amount of reviews for the product with the most reviews.
    """

    reviews_per_product = get_reviews_per_product(input_file)
    max_reviews = reviews_per_product.max()
    product_id = reviews_per_product.idxmax()

    return product_id, max_reviews

def get_product_with_n_reviews(input_file,n):
    """
    From a parsed csv file, returns the productId that has n reviews.
       
    Args:
        input_file (str): The path to the input text file.
    
    Returns: 
        product_id (str): The productId with the most reviews.
    """
    
    reviews_per_product = get_reviews_per_product(input_file)

    # get the reviews of the product that has n reviews
    product_id = reviews_per_product[reviews_per_product == n].index[0]

    return product_id
    

def get_products_with_reviews_in_range(input_file, n, x, y):
    """
    From a parsed CSV file, returns a list of productIds that have between x and y reviews.

    Args:
        input_file (str): The path to the input CSV file.
        n (int): The number of productIds to return.
        x (int): Minimum number of reviews a product must have.
        y (int): Maximum number of reviews a product must have.

    Returns:
        product_ids (list): A list of productIds with between x and y reviews, up to n products.
    """
    
    reviews_per_product = get_reviews_per_product(input_file)
    
    filtered_products = reviews_per_product[(reviews_per_product >= x) & (reviews_per_product <= y)]
    
    product_ids = filtered_products.index[:n].tolist()
    
    return product_ids

def detect_emojis(text):
    """
    Detects emojis in a string.

    Args:
        text (str): The input text that may contain emojis.

    Returns: 
        tue or false: returns 1 if there are emojis in the text,  otherwise.
    """
    return any(char in emoji.EMOJI_DATA for char in text)

# are there emojis in the product with most reviews?

#dataset = pd.read_csv('../data/parsed_input_file.csv')

#reviewString = dataset['reviewText']

#reviewString['emoji_count'] = reviewString.apply(detect_emojis)

#print(reviewString['emoji_count'].value_counts())

nlp = spacy.load('en_core_web_sm')

def tokenize_reviews_to_sequences(reviews, sequence_length):
    """
    Tokenizes reviews into sequences of a given length.
    
    Args:
        reviews (pd.Series): A pandas series containing review texts.
        sequence_length (int): The length of each sequence.

    Returns:
        pd.Series
    """
    sequences = []
    
    for review in reviews:
        doc = nlp(review)
        tokens = [token.text for token in doc if not token.is_punct and not token.is_space]

        for i in range(len(tokens)):
            sequence = ' '.join(tokens[i:i + sequence_length])
            sequences.append(sequence)
    
    return pd.Series(sequences)

# break reviews into a list of sentences
def split_into_sentences(reviews):
    """
    Receives a series of reviews, and returns a series of sentences.
    
    Args:
        reviews (pd.Series): A pandas series containing review texts.

    Returns:
        pd.Series: A series of sentences.
    """
    sentences = []
    
    for review in reviews:
        doc = nlp(review)
        for sent in doc.sents:
            sentences.append(sent.text)
    
    return pd.Series(sentences)

def lemmatize_text(text):
    """
    Lemmatizes a text without removing stopwords.
    
    Args:
        text (str)

    Returns:
        str:
    """
    doc = nlp(text)
    lemmatized_text = ' '.join([token.lemma_ for token in doc if not token.is_punct and not token.is_space])
    return lemmatized_text

def remove_stopwords(text):
    """
    Removes stopwords from the text without lemmatization.
    
    Args:
        text (str)

    Returns:
        str:
    """
    doc = nlp(text)
    no_stopwords_text = ' '.join([token.text for token in doc if not token.is_stop and not token.is_punct and not token.is_space])
    return no_stopwords_text

def lemmatisation_stopwords_series(df, rm_stopwords, lemmatize):
    """
    Lemmatizes and removes stopwords from a pandas series of text.

    Args:
        df (pd.Series): A pandas series of text.
        rm_stopwords (bool): Whether to remove stopwords.
        lemmatize (bool): Whether to lemmatize the text.

    Returns:
        pd.Series: The processed text.
    """

    if rm_stopwords and lemmatize:
        df = df.apply(lemmatize_text)
        df = df.apply(remove_stopwords)
    elif rm_stopwords and not lemmatize:
        df = df.apply(remove_stopwords)
    elif lemmatize and not rm_stopwords:
        df = df.apply(lemmatize_text)

    return df

# --------------------------------------------- 

def get_reviews_from_top_x_products(file_path, x):

    """
    Gets the reviews for the top x products with the most reviews.
    
    Args:
        file_path (str): The path to the input CSV file.
        x (int): The number of products to consider

    Returns:
        pd.Series: The combined reviews for the top x products.
    """
    dataset = pd.read_csv(file_path)
    
    # Contar las reseñas por producto
    product_counts = dataset['productId'].value_counts()
    
    # Obtener los primeros x productos con más reseñas
    top_products = product_counts.head(x).index.tolist()
    
    # Filtrar las reseñas para estos productos y combinarlas
    combined_reviews = dataset[dataset['productId'].isin(top_products)]['reviewText']
    
    return combined_reviews

def filter_nouns_spacy(text_series):
    """
    Filters out non-noun words from a series of text reviews using spaCy.
    
    Args:
        text_series (pd.Series): A pandas series containing review

    Returns:
        pd.Series: A new series with the filtered text
    """
    filtered_text = []
    
    for review in text_series:
        # Procesar el texto con spaCy para obtener las categorías gramaticales
        doc = nlp(review)
        
        # Filtrar palabras que no son sustantivos (pos_ == 'NOUN' para sustantivos)
        non_nouns = [token.text for token in doc if token.pos_ != 'NOUN']
        
        # Unir las palabras filtradas en una cadena de texto
        filtered_text.append(" ".join(non_nouns))
    
    return pd.Series(filtered_text)

def map_lda_to_general_topic(lda_topics, general_topics):

    """
    Maps LDA topics to general topics based on keywords.

    Args:
        lda_topics (list): A list of topics generated by LDA.
        general_topics (dict): A dictionary mapping general topics to keywords.

    Returns:
        dict: A mapping of LDA topics to general topics.
    """
    mapped_topics = {}
    for idx, topic_words in enumerate(lda_topics):
        for general_topic, keywords in general_topics.items():
            if any(any(keyword in word for word in topic_words) for keyword in keywords):
                mapped_topics[f"Topic {idx}"] = general_topic
                break
        else:
            mapped_topics[f"Topic {idx}"] = 'Other'
    return mapped_topics

# vectorization
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation


def optimal_topic_number(sequences, topic_range):
    perplexities = []

    vectorizer = CountVectorizer()

    for num_topics in topic_range:
        lda_model = LatentDirichletAllocation(n_components=num_topics, random_state=0)
        dtm = vectorizer.fit_transform(sequences)
        lda_model.fit(dtm)
        perplexity = lda_model.perplexity(dtm)
        perplexities.append(perplexity)

    optimal_num_topics = topic_range[perplexities.index(min(perplexities))]
    return optimal_num_topics

def filter_adjectives_spacy(text_series):
    filtered_text = []
    
    for review in text_series:
        # Procesar el texto con spaCy para obtener las categorías gramaticales
        doc = nlp(review)
      
        # Filtrar palabras que son adjetivos (pos_ == 'ADJ' para adjetivos)
        non_adjectives = [token.text for token in doc if token.pos_ != 'ADJ']
       
        # Unir los adjetivos en una cadena de texto
        filtered_text.append(" ".join(non_adjectives))
   
    return pd.Series(filtered_text)

from collections import Counter

def remove_frequent_words(reviews, top_n_frequent):
    """
    Remove the most frequent words from the reviews.

    Args:
        reviews : pd.Series
        top_n_frequent : int, number of frequent words to remove

    Returns:
        review_cleaned : pd.Series
    """

    # Tokenize the reviews into words
    all_words = reviews.str.split().explode()

    # Find the most frequent words
    most_common_words = [word for word, count in Counter(all_words).most_common(top_n_frequent)]

    # Remove the most frequent words from the reviews
    reviews_cleaned = reviews.apply(lambda review: ' '.join([word for word in review.split() if word not in most_common_words]))

    return reviews_cleaned

def clean_reviews(reviews, params):
    """
    Preprocess the reviews

    Args:
        review : pd.Series
        params : dict
    
    Returns:
        review_cleaned : pd.Series
    """

    # handling missing values
    if params['nan']:
        reviews = reviews.dropna()

    # removing emojis
    if params['emojis']:
        reviews = reviews.apply(lambda x: emoji.replace_emoji(x, replace=''))

    # lowercase the reviews 
    if params['lowercase']:
        reviews = reviews.str.lower()

    # remove extra white-spaces
    if params['whitespaces']:
        reviews = reviews.str.strip().str.replace(r'\s+', ' ', regex=True)

    # remove special characters
    #if params['special_chars']:
    #    reviews = reviews.str.replace(r'[^\w\s.,!?]', '', regex=True)

    # remove numbers
    if params['numbers']:
        reviews = reviews.str.replace(r'\d+', '', regex=True)
    
    # remove urls and email addresses
    if params['emails_and_urls']:
        reviews = reviews.str.replace(r'http\S+|www\S+|mailto:\S+', '', regex=True)
        reviews = reviews.str.replace(r'\S+@\S+', '', regex=True) 

    # transform contractions
    if params['contractions']:
        reviews = reviews.apply(contractions.fix)

    # remove nouns
    if params['nouns']:
        reviews = filter_nouns_spacy(reviews)

    # remove adjectives
    if params['adj']:
        reviews = filter_adjectives_spacy(reviews)

    # remove k frequent words
    if params['most_frequent']>0:
        reviews = remove_frequent_words(reviews, params['most_frequent'])

    return reviews

