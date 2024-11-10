import ast
import os
import pandas as pd
from transformers import pipeline

os.environ["HUGGINGFACE_HUB_TOKEN"] = "KEY"

def summary_gen():
    """
    Generate a summary of the topics and the sentiment analysis
    """

    # load sentences w topic and sentiment
    base_dir = os.path.dirname(os.path.abspath(__file__))
    processed_dir = os.path.join(base_dir, '../../data/processed/sequences_topics_sentiment.csv')
    data = pd.read_csv(processed_dir)
    
    # words in each topic
    topic_info_path = os.path.join(base_dir, '../../data/processed/topic_info.csv')
    topic_representation = pd.read_csv(topic_info_path)

    # first part of the prompt
    prompt = (
        "The following are topics detected in reviews by users. Give me a summary of what the reviews "
        "say about the product using the following information:\n\n"
    )

     # construct the final prompt using the topics representative words and the most positive and negative sequences
    for topic_id in data['topic_id'].unique():
        # filtering
        topic_data = data[data['topic_id'] == topic_id]
        topic_keywords_str = topic_representation.loc[topic_representation['Topic'] == topic_id].iloc[0, 3]
        topic_keywords = ast.literal_eval(topic_keywords_str) if isinstance(topic_keywords_str, str) else topic_keywords_str
        
        # get the most positive and most negative sentences
        positive_sequence = topic_data.loc[topic_data['sentiment'].idxmax()]['sequence']
        negative_sequence = topic_data.loc[topic_data['sentiment'].idxmin()]['sequence']

        # add each topic info to the prompt
        prompt += (
            f"* Topic {topic_id}. Representative words = {', '.join(topic_keywords)}\n"
            f"  - Most negative sentence: {negative_sequence}\n"
            f"  - Most positive sentence: {positive_sequence}\n\n"
        )

    # generate the summary with a single call to the model
    model = pipeline("text-generation", model="meta-llama/Llama-2-7b-hf") 
    summary = model(prompt, max_length=500, do_sample=False)
    print(summary[0]["generated_text"])

    return 0


def average_sentiment():
    """
    Calculate the average sentiment for each topic
    """

    base_dir = os.path.dirname(os.path.abspath(__file__))
    processed_dir = os.path.join(base_dir, '../../data/processed/sequences_topics_sentiment.csv')

    data = pd.read_csv(processed_dir)

    avg_sentiment = data.groupby('topic_id')['sentiment'].mean()

    return avg_sentiment

summary_gen()