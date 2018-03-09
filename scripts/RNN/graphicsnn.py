from os import listdir
import os
import json
import spacy   
import numpy as np        
import sys
import linecache
import time
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.models import load_model
from sklearn.metrics import confusion_matrix

###################################################
def compute_cm(y_test,y_predicted): # function that computes confusion matrix, receives np arrays
	true_labels=np.argmax(y_test,axis=-1)
	predicted_labels=np.argmax(y_predicted,axis=-1)
	CFM=confusion_matrix(true_labels, predicted_labels)
	NCFM=CFM.astype('float') / CFM.sum(axis=1)[:, np.newaxis]
	return CFM,NCFM
###################################################

dire='/vol/work/mesa/firstnnsoftmax/'
emblist=list(range(2,11))

for emb in emblist:
	dirext=str(emb)+'.embeding/'
	bestacc=0
	for folder in listdir(dire+dirext):
		if 'flayer' in folder:
			folders=folder.split('.')
			neuronsnumber=folders[1]
			datadev=np.load(dire+dirext+folder+'/devaccuracy.npy')
			accuracy=np.amax(datadev)
			if accuracy>bestacc: #we keep the best accuracy and best neuronnumber
				bestacc=accuracy
				bestneuron=neuronsnumber
	
	##we save the best results for each context
	#f=open('/vol/work/mesa/bestneuronsnvscontext','a')
	#f.write(str(emb)+' '+str(bestneuron)+' '+str(bestacc)+'\n')
	#f.close()	
	
	#now we plot training and dev accuracy on bestneuron
	datatrain=np.load(dire+dirext+'flayerneurons.'+str(bestneuron)+'/trainaccuracy.npy')
	datadev=np.load(dire+dirext+'flayerneurons.'+str(bestneuron)+'/devaccuracy.npy')
	xaxis=list(range(1,len(datatrain)+1))
	plt.plot(xaxis,datatrain,xaxis,datadev)
	plt.grid()
	plt.title('Accuracy on training, single layer NN '+str(emb)+' context')
	plt.ylabel('accuracy')
	plt.xlabel('epoch')
	#plt.xlim(0.0, 100.0)
	legendlist=['train','max dev accuracy = '+str(bestacc)]
	plt.legend(legendlist, loc='best')
	plt.savefig('/people/mesa/Desktop/nn'+str(emb)+'.png')
	#plt.clf()
	#plt.cla()
	#plt.close()
	#we save confusion matrix per each context
	
	#Xtest=np.load('/vol/work/mesa/wemb/Xtest'+str(emb)+'.npy')
	#Ytest=np.load('/vol/work/mesa/wemb/Ytest'+str(emb)+'.npy')
	#trainedmodel = load_model(dire+dirext+'flayerneurons.'+str(bestneuron)+'/weights.hdf5')
	#Ypredict=trainedmodel.predict(Xtest, batch_size=32, verbose=0)
	#[CFM,NCFM]=compute_cm(Ytest,Ypredict)
	#print(NCFM)
	#input()
#plt.show()
