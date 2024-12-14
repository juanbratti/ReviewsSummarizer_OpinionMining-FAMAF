from gensim.models import CoherenceModel
from gensim.corpora import Dictionary
from sklearn.metrics import silhouette_score
import numpy as np
import os
import pandas as pd
from .scripts.preprocessing import load_and_preprocess_data, tokenize_reviews, vectorize_sequences
from .scripts.topic_modelling import apply_bertopic, return_topic_sequences_csv, visualize, print_model_info


def get_scores(topic_model, umap_model, embeddings, topics, sequences_list):

    # Coherence Model
    topics_cm = topic_model.get_topics()
    topic_words = [
        [word for word, _ in topic]
        for _, topic in topics_cm.items()
        if topic and any(word.strip() for word, _ in topic)
        ]
    sequences = [sequence.split() for sequence in sequences_list]
    dictionary = Dictionary(sequences)
    coherence_model = CoherenceModel(topics=topic_words,
                                     texts=sequences,
                                     dictionary=dictionary,
                                     coherence="c_v")
    coherence = coherence_model.get_coherence()

    # Silhouette Score
    umap_embeddings = topic_model.umap_model.transform(embeddings)
    indices = [index for index, topic in enumerate(topics) if topic != -1]
    try:
        X = umap_embeddings[np.array(indices)]
        labels = [topic for index, topic in enumerate(topics) if topic != -1]
        score_silhouette = silhouette_score(X, labels)
    except:
        score_silhouette = 0        

    return (coherence, score_silhouette)

def run_tm(params, category=""):

    base_dir = os.path.dirname(os.path.abspath(__file__))
    processed_dir = os.path.join(base_dir, '../data/processed')

    reviews_cleaned = pd.read_csv(os.path.join(processed_dir, f'size_{params["product_review_count"]}_processed{category}.csv'))['reviewText']

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
    
    scores = get_scores(topic_model, umap_model, embeddings, topics, sequences_list)
    return scores


def review_count(fixed_tokens, review_counts):
    params = {
        'new_reviews': 0,  # 0 for old reviews, 1 new reviews
        'product_review_count': 10, #596,
        'tokens': fixed_tokens,
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
    scores_list = []
    for r in review_counts:
        print(f"Running for Review Count: {r}")
        params["product_review_count"] = r
        scores_list.append(run_tm(params))
    return scores_list

def n_tokens(fixed_review, tokens_counts):
    params = {
        'new_reviews': 0,  # 0 for old reviews, 1 new reviews
        'product_review_count': fixed_review, #596,
        'tokens': 0,
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
    scores_list = []
    for t in tokens_counts:
        print(f"Running for Token Count: {t}")
        params["tokens"] = t
        scores_list.append(run_tm(params))
    return scores_list

def aditional_params(fixed_tokens, fixed_reviews, most_frequent):
    params = {
        'new_reviews': 0,  # 0 for old reviews, 1 new reviews
        'product_review_count': fixed_reviews, #596,
        'tokens': fixed_tokens,
        # delete?
        'nan': True, 
        'emojis': True,
        'contractions': True,
        'special_chars': True,
        'whitespaces': True,
        'stopwords': False,
        'lemmatization': True,
        'lowercase': True,
        'emails_and_urls': True,
        'nouns': False,
        'adj': False,
        'numbers': False,
        'most_frequent': 0
    }
    scores_list = []
    params_to_variate = ["raw", "stopwords", "nouns", "adj", "numbers", "most_frequent"]
    for p in params_to_variate:
        print(f"Running for Param: {p}")
        if p != "most_frequent":
            params[p] = True
        else:
            params[p] = most_frequent
        if p == "raw":
            scores_list.append(run_tm(params))
        else:
            scores_list.append(run_tm(params, "_" + p))
        if p != "most_frequent":
            params[p] = False
        else:
            params[p] = 0
        
    return scores_list

from matplotlib import pyplot as plt
import numpy as np

def graph_scores(x, scores, x_label, title):
    y1 = []
    y2 = []
    for score in scores:
        y1.append(score[0])
        y2.append(score[1])
    
    plt.figure(figsize=(10, 6), dpi=200)
    plt.plot(x, y1, marker="o", label="Coherence")
    plt.plot(x, y2, marker="o", label="Silhouette")
    plt.xlabel(x_label)
    plt.ylabel("Scores")
    plt.title(title)
    plt.legend(title="Metric")
    plt.savefig(f"src/images/evaluation_{x_label}.png", dpi=200)
    plt.close()
    return

def bar_graph(scores_ap):
    y1 = []
    y2 = []
    for score in scores_ap:
        y1.append(score[0])
        y2.append(score[1])


    scores_categories = ["raw", "stopwords", "nouns", "adj", "numbers", "most_frequent"]
    metrics = ["Coherence", "Silhouette"]

    scores = {
        "Coherence": y1,
        "Silhouette": y2
    }

    x = np.arange(len(metrics))
    bar_width = 0.15
    offset = np.arange(len(scores_categories)) * bar_width - (len(scores_categories) - 1) * bar_width / 2


    fig, ax = plt.subplots(figsize=(10, 6))
    for i, c in enumerate(scores_categories):
        ax.bar(x + offset[i], [scores[m][i] for m in metrics], width=bar_width, label=c)

    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel("Score")
    ax.set_title("Scores for deletion of set parameters")
    ax.legend(title="Categories")
    plt.tight_layout()
    plt.savefig(f"src/images/evaluation_params.png", dpi=200)
    plt.close()
    return 

# pre: reviews for: 10, 20, 30, 40, 49


# review_counts = [10, 20, 30, 40, 49]
# fixed_tokens = 7
# print("Running for Review Counts...")
# scores_list = review_count(fixed_tokens, review_counts)
# x_label = "Review Count"
# title = "Score basado en Reviews Counts"
# graph_scores(review_counts, scores_list, x_label, title) # (x, y)

# tokens_counts = [1, 3, 5, 7, 9]
# fixed_review = 20
# print("Running for Tokens Counts...")
# scores_list = n_tokens(fixed_review, tokens_counts)
# x_label = "Tokens"
# title = "Score basado en N Tokens"
# graph_scores(tokens_counts, scores_list, x_label, title) # (x, y)


fixed_reviews = 10
fixed_tokens = 3
most_frequent = 5
print("Running for Aditional Params...")
scores_ap = aditional_params(fixed_tokens, fixed_reviews, most_frequent)
bar_graph(scores_ap)

# Cantidad de reseñas

# Tokens

# Stopwords si stopwords no creo que no va a causar problema porque se van al -1

# Adjetivos, sustantivos
