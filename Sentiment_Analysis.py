#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from transformers import pipeline

def get_sentiment(text):
    """
    Implements sentiment analysis on text
    output will be a list of dictionaries containing 
    labels and scores
    """
    distilled_student_sentiment_classifier = pipeline(
        model = 'mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis',
        return_all_scores=True
    )

    output = distilled_student_sentiment_classifier (text)
    # print(output)
    
    highest_score = 0
    highest_label = None

    # Iterate over the output list
    for item in output[0]:
        # If the score of the current item is higher than the highest score
        if item['score'] > highest_score:
            # Update the highest score and label variables
            highest_score = item['score']
            highest_label = item['label']
    return highest_label

