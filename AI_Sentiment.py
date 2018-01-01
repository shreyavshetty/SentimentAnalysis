#import the required library
import os
import sys
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Dense, Input, GlobalMaxPooling1D
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model
import random
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Embedding
from keras.layers import LSTM , Bidirectional,Dropout
import json

max_features = 20000 #corpus size
maxlen = 500 # cut texts after this number of words (among top max_features most common words)
batch_size = 128	#the number of training examples in one forward/backward pass.
textprocessing =[]
textprocessing1 =[]
train = []
test = []
with open('pos_amazon_cell_phone_reviews.json') as jsondata:
	d=json.load(jsondata)#loads jsons as dictionaries
	jsondata.close()

l=d[u'root']
for j in range(0,len(l)):
	text1=[]
	text1.append(1)
	text1.append(l[j][u'summary'])#fectch the summary
	textprocessing.append(text1)#append the summary to a list
index = random.sample(range(0, len(textprocessing)), 10000)
for i in index:
	if i < len(textprocessing) :
		train.append(textprocessing[i])#obtain training data as a list
		del textprocessing[i]

index = random.sample(range(0, len(textprocessing)),650)
for i in list(index):
	test.append(textprocessing[i])#obtain testing data as a list

with open('neg_amazon_cell_phone_reviews.json') as jsondata:
	d=json.load(jsondata)
	jsondata.close()
l =d[u'root']
for j in range(0,len(l)):
	text2 = []
	text2.append(0)
	text2.append(l[j][u'summary'])	
	textprocessing1.append(text2)

index = random.sample(range(0, len(textprocessing1)), 10000)
for i in list(index):
	if i < len(textprocessing1) :
		train.append(list(textprocessing1[i]))
		del textprocessing1[i]

index = random.sample(range(0, len(textprocessing1)), 650)
for i in list(index):
	test.append(textprocessing1[i])


x_train =[]
x_test =[]
y_train =[]
y_test =[]
for i in train:
	x_train.append(i[1])
	y_train.append(i[0])
for i in test:
	x_test.append(i[1])
        y_test.append(i[0])


# finally, vectorize the text samples into a 2D integer tensor
tokenizer = Tokenizer() #instantiate the tokenizer
tokenizer.fit_on_texts(x_train)#fit the given trained text
sequences = tokenizer.texts_to_sequences(x_train)#list of sequences (one per text input).

tokenizer2 = Tokenizer()
tokenizer2.fit_on_texts(x_test)#fit the given test text
sequences2 = tokenizer2.texts_to_sequences(x_test)#list of sequences (one per text input).

x_train = sequence.pad_sequences(sequences, maxlen=maxlen)#Transform a list of train sequences (lists of scalars) into a 2D Numpy array of shape
x_test = sequence.pad_sequences(sequences2, maxlen=maxlen)#Transform a list of test sequences (lists of scalars) into a 2D Numpy array of shape

#print x_train, x_test shape
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

#If your training data uses classes as numbers, to_categorical will transform those numbers in proper vectors for using with models
Y_train = to_categorical(y_train)
Y_test = to_categorical(y_test)

#Build models
print('Build model ~')
model = Sequential()#sequence 
model.add(Embedding(max_features, 32))#word embedding
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))#LSTM network
model.add(Dense(2, activation='sigmoid'))#add output layer


# try using different optimizers and different optimizer configs
model.compile(loss='binary_crossentropy', 
              optimizer='adam',
              metrics=['accuracy'])

print('Train ~')
model.fit(x_train, Y_train,validation_split=0.16,
          batch_size=batch_size,
          epochs=2)
score , acc = model.evaluate(x_test, Y_test,
                            batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc) 
model.summary()
