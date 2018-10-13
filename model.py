# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 22:31:05 2018

@author: rohith
"""

import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Embedding
from keras.utils import to_categorical

dataframe_train = pd.read_csv('train.tsv', delimiter = '\t')
dataframe_test = pd.read_csv('test.tsv', delimiter = '\t')

sentences_train_x = np.asarray(dataframe_train['Phrase'])
sentences_train_y = to_categorical(np.asarray(dataframe_train['Sentiment']))
sentences_test_x = np.asarray(dataframe_test['Phrase'])
#sentences_test_y = np.asarray(dataframe_test['Sentiment'])

sentences_x = np.concatenate((sentences_train_x, sentences_test_x) , axis = 0)
t = Tokenizer()
t.fit_on_texts(sentences_x)

#print(t.word_index)
vocab_length = len(t.word_index)+1

glove_lookup = {}
encoded_texts_train = t.texts_to_sequences(sentences_train_x)
encoded_texts_test = t.texts_to_sequences(sentences_test_x)
# pad documents to a max length of 4 words
max_length = 10
padded_texts_train = pad_sequences(encoded_texts_train, maxlen=max_length, padding='post')
padded_texts_test = pad_sequences(encoded_texts_test, maxlen=max_length, padding='post')
#print(padded_texts_train)

f = open('./glove.6B.50d.txt','r', encoding = 'utf-8')

for line in f:
    values = line.split()    
    word = values[0]
    vector = values[1:]
    glove_lookup[word] = vector

embedding_matrix = np.zeros((vocab_length, 50))
for word,i in t.word_index.items():
    if word in glove_lookup:
        vector = glove_lookup[word]
    else:
        vector = np.zeros(50)
    embedding_matrix[i] = vector

f = open('./train.tsv')
model = Sequential()

e = Embedding(vocab_length, 50 , weights = [embedding_matrix], input_length = max_length, trainable = False)
model.add(e)
model.add(Dense(20, activation = 'relu'))
model.add(Flatten())
model.add(Dense(5, activation = 'softmax'))

model.compile(optimizer = 'adam' , loss = 'binary_crossentropy' , metrics = ['acc'])

model.fit( padded_texts_train,sentences_train_y , epochs = 100, verbose = 2)

loss,accuracy = model.evaluate(padded_texts_train, sentences_train_y, verbose = 0)

model.save('sentimentclassification.h5')

output_test = np.argmax(model.predict(padded_texts_test),axis = 1)



