# sentiment analysis with textblob
import os
import pandas as pd
from textblob import TextBlob

def sentiment_analysis(source):

    base_dir = os.path.dirname(os.path.abspath(__file__))
    processed_dir = os.path.join(base_dir, '../../data/processed')

    sentiment = source['sequence'].apply(lambda x: TextBlob(x).sentiment.polarity if isinstance(x, str) else 0)

    df = pd.DataFrame({
        'sequence': source['sequence'],
        'topic_id': source['topic_id'], 
        'sentiment': sentiment
    })

    df.to_csv(os.path.join(processed_dir,'sequences_topics_sentiment.csv'), index=False)

    return df
