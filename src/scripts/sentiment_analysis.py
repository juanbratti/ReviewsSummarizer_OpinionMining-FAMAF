# sentiment analysis with textblob

import pandas as pd
from textblob import TextBlob

def sentiment_analysis(source):

    sentiment = source['sequence'].apply(lambda x: TextBlob(x).sentiment.polarity if isinstance(x, str) else 0)

    df = pd.DataFrame({
        'sequence': source['sequence'],
        'topic_id': source['topic_id'],  # Asegúrate de que esta columna exista en tu CSV
        'sentiment': sentiment
    })

    df.to_csv('../data/processed/sequences_topics_sentiment.csv', index=False)

    return df
