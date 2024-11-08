import os
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from bertopic.vectorizers import ClassTfidfTransformer
import pandas as pd

def apply_bertopic(sequences, model, reduced_topics):
    """
    Apply BERTopic to extract topics from the reviews using seed topic list.

    Args:
        sequences (list): list/series of tokenized sequences 
        seed_topic_list (list): list of seed topics to guide the model.

    Returns:
        topic_model (BERTopic): Trained BERTopic model.
        topics (list): list of topics found in the sequences.
    """
    embedding_model = SentenceTransformer(model)
    embeddings = embedding_model.encode(sequences, show_progress_bar=True)
    
    umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine", random_state=42)
    hdbscan_model = HDBSCAN(min_cluster_size=15, metric='euclidean', cluster_selection_method='eom', prediction_data=True)
    vectorizer_model = CountVectorizer()
    ctfidf_model = ClassTfidfTransformer()

    topic_model = BERTopic(umap_model=umap_model, embedding_model=embedding_model, hdbscan_model=hdbscan_model, vectorizer_model=vectorizer_model, ctfidf_model=ctfidf_model)

    topics, _ = topic_model.fit_transform(sequences, embeddings)
    
    topic_model.reduce_topics(sequences, nr_topics=reduced_topics)
    
    # similar_topics, similarity = topic_model.find_topics("price", top_n=5)
    # print(topic_model.get_topic(similar_topics[0]))

    return topic_model

def visualize(topic_model, sequences_list, model):
    fig = topic_model.visualize_topics()
    fig.write_image(f"./images/bt_topics_{model}.png")
    fig = topic_model.visualize_hierarchy()
    fig.write_image(f"./images/bt_topic_hierarchy_{model}.png")
    fig = topic_model.visualize_barchart()
    fig.write_image(f"./images/bt_topic_barchart_{model}.png")
    fig = topic_model.visualize_heatmap()
    fig.write_image(f"./images/bt_topic_similarity_heatmap_{model}.png")
    fig = topic_model.visualize_term_rank()
    fig.write_image(f"./images/bt_term_score_decline_{model}.png")

    # topic_distr, _ = topic_model.approximate_distribution(sequences_list)
    # # Elegir el doc que uno quiera y muestra la distribucion de prob
    # fig = topic_model.visualize_distribution(topic_distr[1])
    # fig.write_image(f"./images/bt_topic_dist_{model}.png")
    return

def print_model_info(topic_model, sequences_list, model):
    print("model:"+model)
    print("Topics discovered:")
    print(topic_model.get_topic_info())
    print("-----------------------------------------")
    print("Document info: ")
    # print(topic_model.get_document_info(sequences_list))
    # print(topic_model.get_representative_docs())
    print("-----------------------------------------")
    # print(topic_model.topic_labels_)
    # print("Heriarchical topics:")
    # print(topic_model.hierarchical_topics(sequences_list))
    # print(topic_model.topic_aspects_)
    
    print("-----------------------------------------")
    return

def return_topic_sequences_csv(model, sequences_series):
    # Obtener los IDs de los temas asignados durante el fit_transform
    topic_ids = model.topics_

    # Filtrar solo las secuencias con temas asignados (topic_id != -1)
    filtered_sequences = []
    filtered_topic_ids = []

    for sequence, topic_id in zip(sequences_series, topic_ids):
        if topic_id != -1:
            filtered_sequences.append(sequence)
            filtered_topic_ids.append(topic_id)

    # Crear el DataFrame con las secuencias filtradas y sus topic_ids
    result_df = pd.DataFrame({
        'sequence': filtered_sequences,
        'topic_id': filtered_topic_ids
    })
    
    # Guardar el DataFrame en un archivo CSV
    base_dir = os.path.dirname(os.path.abspath(__file__))
    processed_dir = os.path.join(base_dir, '../../data/processed')
    result_df.to_csv(os.path.join(processed_dir,'sequences_topics.csv'), index=False)

    return result_df
