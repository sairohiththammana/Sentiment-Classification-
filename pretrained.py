# -*- coding: utf-8 -*-
"""
Created on Sat Oct 13 13:14:07 2018

@author: rohith
"""
#0 - negative
#1 - somewhat negative
#2 - neutral
#3 - somewhat positive
#4 - positive
from keras.models import load_model
import numpy as np
import pickle as pk
def sentiment(x):
    number_to_sentiment_mapping = ["Negative" , "Somewhat Negative" , "Neutral" , "Somewhat postive" , "positive"]
    model = load_model('sentimentclassification.h5')     
    with open('tokenizer.pickle', 'rb') as handle:
        t = pk.load(handle) 
    x = np.asarray([x])
    t.oov_token = None
    encoded_texts_test = t.texts_to_sequences(x)
    padded_texts_test = pad_sequences(encoded_texts_test, maxlen=10, padding='post')
    sentiment_index = np.argmax(model.predict(padded_texts_test),axis = 1)[0]
    return number_to_sentiment_mapping[sentiment_index]
    
    
    


if __name__ == "__main__":
    print(sentiment("'A series of escapades leading to bad experience worse"))
    