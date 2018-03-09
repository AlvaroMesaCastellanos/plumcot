from keras.models import Sequential
from keras.layers import LSTM,Bidirectional
from keras.layers import Dense, Activation
from sklearn.metrics import confusion_matrix
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from os import listdir
import keras
import os
import json
import spacy   
import numpy as np        
import sys
import linecache
import time
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
#script that performs LSTM train and test, main purpose is to understand its performance
epochsnumber=4000 #number of epochs to perform training
neuronsn=int(os.environ['SGE_TASK_ID']) #we put a step of 5
#neuronsn=sys.argv[1]
neuronsn=pow(2,neuronsn) #we perform 2^neuronsn, 1,2,4,16...,512,until 1024
embedingslist=list(range(1,11)) # we perfom training from 2-10 context (x2)
weights={0: 0.925,1:0.5981,2:0.5981,3:0.4774}
for embeding in embedingslist:
	# Compile model
	model=Sequential()
	model.add(Bidirectional(LSTM(int(neuronsn), return_sequences=False),input_shape=(2*embeding, 300),merge_mode='concat'))
	model.add(Dense(4, activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	# Load X,Y and Fit the model
	X=np.load('/vol/work/mesa/wemb4classes/Xtrain'+str(embeding)+'.npy')
	Y=np.load('/vol/work/mesa/wemb4classes/Ytrain'+str(embeding)+'.npy')
	Xdev=np.load('/vol/work/mesa/wemb4classes/Xdev'+str(embeding)+'.npy')
	Ydev=np.load('/vol/work/mesa/wemb4classes/Ydev'+str(embeding)+'.npy')
	#Reshape according to LSTM
	dim=X.shape
	X=X.reshape(dim[0],2*embeding,300)
	dim=Xdev.shape
	Xdev=Xdev.reshape(dim[0],2*embeding,300)
	# Train the model with train and create folder to store the models
	path='/vol/work/mesa/LSTMearlystoppingweights4classes/'+str(embeding)+'.embeding/'+'flayerneurons.'+str(neuronsn)
	os.system("mkdir "+path)
	checkpointer = ModelCheckpoint(filepath=path+'/weights.hdf5',monitor='val_acc',verbose=1, save_best_only=True,mode='max') ##{epoch:02d}.
	checkpointer2=EarlyStopping(monitor='val_acc',min_delta=0,patience=50,verbose=1,mode='max')
	history=model.fit(X, Y, epochs=int(epochsnumber), batch_size=50,verbose=0,class_weight=weights,validation_data=(Xdev,Ydev),callbacks=[checkpointer,checkpointer2])
	#we save the accuracy of each trained model
	np.save(path+'/trainaccuracy',history.history['acc'])
	np.save(path+'/devaccuracy',history.history['val_acc'])
	del(X)
	del(Y)
	del(Xdev)
	del(Ydev)
