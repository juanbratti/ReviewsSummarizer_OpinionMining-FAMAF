import os
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from bertopic.vectorizers import ClassTfidfTransformer
import pandas as pd
from wordcloud import WordCloud
import math
import matplotlib.pyplot as plt


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


    

    create_wordcloud_grid(topic_model)

    # create csv with the topic id and the words in the topic
    topic_info = topic_model.get_topic_info()
    topic_info_df = pd.DataFrame(topic_info)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    processed_dir = os.path.join(base_dir, '../../data/processed')
    topic_info_df.to_csv(os.path.join(processed_dir, 'topic_info.csv'), index=False)

    return topic_model, umap_model, embeddings, topics

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

def create_wordcloud_grid(topic_model):
    # Get all topics from the model
    base_dir = os.path.dirname(os.path.abspath(__file__))
    processed_dir = os.path.join(base_dir, '../images')

    save_path = os.path.join(processed_dir, "wordcloud_grid.png")

    topics = topic_model.get_topics().keys()
    num_topics = len(topics)
    
    grid_size = math.ceil(math.sqrt(num_topics)) 
    
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))
    fig.subplots_adjust(hspace=0.3, wspace=0.3)
    
    for idx, topic in enumerate(topics):
        text = {word: value for word, value in topic_model.get_topic(topic)}
        wc = WordCloud(background_color="white", max_words=1000)
        wc.generate_from_frequencies(text)
        
        ax = axes[idx // grid_size, idx % grid_size]
        ax.imshow(wc, interpolation="bilinear")
        ax.set_title(f"Topic {topic}", fontsize=10)
        ax.axis("off")
    
    for j in range(idx + 1, grid_size ** 2):
        fig.delaxes(axes[j // grid_size, j % grid_size])
    
    plt.savefig(save_path, format="png")
