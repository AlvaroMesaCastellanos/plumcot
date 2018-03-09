#script that trains and test a basic neural network
from keras.models import Sequential
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

###################################################
def compute_cm(y_test,y_predicted): # function that computes confusion matrix, receives np arrays
	true_labels=np.argmax(y_test,axis=-1)
	predicted_labels=np.argmax(y_predicted,axis=-1)
	CFM=confusion_matrix(true_labels, predicted_labels)
	NCFM=CFM.astype('float') / CFM.sum(axis=1)[:, np.newaxis]
	return CFM,NCFM
###################################################

epochsnumber=4000 #number of epochs to perform training
neuronsn=int(os.environ['SGE_TASK_ID'])
#neuronsn=sys.argv[1]
neuronsn=pow(2,neuronsn) #we perform 2^neuronsn, 1,2,4,16...,512,until 1024
embedingslist=list(range(1,11)) # context from 1 to 10
for embeding in embedingslist:
	model = Sequential()
	indim=embeding*600
	model.add(Dense(int(neuronsn), input_dim=indim))
	model.add(Activation('relu'))
	#second layer
	#model.add(Dense(3600, activation='relu'))
	model.add(Dense(4, activation='softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	# Load X,Y and Fit the model
	X=np.load('/vol/work/mesa/wemb4classes/Xtrain'+str(embeding)+'.npy')
	Y=np.load('/vol/work/mesa/wemb4classes/Ytrain'+str(embeding)+'.npy')
	Xdev=np.load('/vol/work/mesa/wemb4classes/Xdev'+str(embeding)+'.npy')
	Ydev=np.load('/vol/work/mesa/wemb4classes/Ydev'+str(embeding)+'.npy')
	# Train the model with train and create folder to store the models
	path='/vol/work/mesa/NNearlystopping4classes/'+str(embeding)+'.embeding/'+'flayerneurons.'+str(neuronsn)
	os.system("mkdir "+path)
	checkpointer = ModelCheckpoint(filepath=path+'/weights.hdf5',monitor='val_acc',verbose=1, save_best_only=True,mode='max') ##{epoch:02d}.
	checkpointer2=EarlyStopping(monitor='val_acc',min_delta=0,patience=50,verbose=1,mode='max')
	history=model.fit(X, Y, epochs=int(epochsnumber), batch_size=50,verbose=0,validation_data=(Xdev,Ydev),callbacks=[checkpointer,checkpointer2])
	#we save the accuracy of each trained model
	np.save(path+'/trainaccuracy',history.history['acc'])
	np.save(path+'/devaccuracy',history.history['val_acc'])
# evaluate the model
#Xtest=np.load('/people/mesa/Desktop/scripts/RNN/wemb/Xtest3.npy')
#Ytest=np.load('/people/mesa/Desktop/scripts/RNN/wemb/Ytest3.npy')
#scores = model.evaluate(Xtest, Ytest)
#Ypredict=model.predict(Xtest, batch_size=32, verbose=0)
#we save the accuracy of trained model
#np.save('/people/mesa/Desktop/scripts/RNN/trainaccuracy',history.history['acc'])
####Print confusion matrix
#[CFM,NCFM]=compute_cm(Ytest,Ypredict)
#print('\n')
#print(NCFM)
####Plot accuracy and loss
#epochslist=list(range(1,int(epochsnumber)+1))
##accuracy
#plt.figure(1)
#plt.scatter(epochslist,historyTrain.history['acc'],s=50)
#plt.scatter(epochslist,historyDev.history['acc'],c='r',s=50)
#plt.grid()


##loss
#plt.scatter(epochslist,history.history['loss'],c='r')
#plt.grid()
#plt.title('model loss')
#plt.ylabel('loss')
#plt.xlabel('epoch')
#plt.legend(['train', 'test'], loc='best')
#plt.show()

