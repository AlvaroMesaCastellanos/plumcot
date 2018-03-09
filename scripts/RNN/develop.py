#script that takes trained models and evaluate them over dev
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, Activation
from sklearn.metrics import confusion_matrix
from keras.callbacks import ModelCheckpoint
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

traindirectory='/people/mesa/Desktop/scripts/RNN/firstnn/'
##we load test and dev data
Xdev=np.load('/people/mesa/Desktop/scripts/RNN/Xdev3.npy')
Ydev=np.load('/people/mesa/Desktop/scripts/RNN/Ydev3.npy')
trainacc=np.load('/people/mesa/Desktop/scripts/RNN/trainaccuracy.npy')
#########################
Xtest=np.load('/people/mesa/Desktop/scripts/RNN/Xtest3.npy')
Ytest=np.load('/people/mesa/Desktop/scripts/RNN/Ytest3.npy')

acclistdev=[]
acclisttest=[]
epochsnumber=[]
epochaux=1
for model in listdir(traindirectory):
	trainedmodel = load_model(traindirectory+model)
	scoresdev = trainedmodel.evaluate(Xdev, Ydev, verbose=0)
	scorestest = trainedmodel.evaluate(Xtest, Ytest, verbose=0)
	acclistdev.append(scoresdev[1])
	acclisttest.append(scorestest[1])
	epochsnumber.append(epochaux)
	epochaux=epochaux+1

plt.plot(epochsnumber,trainacc,epochsnumber,acclistdev,epochsnumber,acclisttest)
plt.grid()
plt.title('dev accuracy over trained model')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','dev', 'test'], loc='best')
#plt.show()
plt.savefig('/people/mesa/Desktop/scripts/RNN/a.pdf')
