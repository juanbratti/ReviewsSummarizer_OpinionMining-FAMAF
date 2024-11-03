from bertopic import BERTopic
from sentence_transformers import SentenceTransformer

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
    
    topic_model = BERTopic()

    topic_model.fit_transform(sequences, embeddings)
    
    topic_model.reduce_topics(sequences, nr_topics=reduced_topics)

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