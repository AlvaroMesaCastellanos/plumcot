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

dire='/vol/work/mesa/NNearlystopping/'
emblist=list(range(1,11))

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
	if False:
		f=open('/vol/work/mesa/NNearlystopping/neuronsvscontext','a')
		f.write(str(emb)+' '+str(bestneuron)+' '+str(bestacc)+'\n')
		f.close()	
	
	#now we plot training and dev accuracy on bestneuron
	datatrain=np.load(dire+dirext+'flayerneurons.'+str(bestneuron)+'/trainaccuracy.npy')
	datadev=np.load(dire+dirext+'flayerneurons.'+str(bestneuron)+'/devaccuracy.npy')
	xaxis=list(range(1,len(datatrain)+1))
	plt.plot(xaxis,datatrain,xaxis,datadev)
	plt.grid()
	plt.title('Accuracy on training, NNearlystopping '+str(emb)+' context, best neuron '+str(bestneuron))
	plt.ylabel('accuracy')
	plt.xlabel('epoch')
	#plt.xlim(0.0, 100.0)
	legendlist=['train','max dev accuracy = '+str(bestacc)]
	plt.legend(legendlist, loc='best')
	plt.savefig('/people/mesa/Desktop/results/NNearlystopping/'+str(emb)+'.png')
	plt.clf()
	plt.cla()
	plt.close()
	#we save confusion matrix per each context
	if True:
		Xtest=np.load('/vol/work/mesa/wemb/Xtest'+str(emb)+'.npy')
		Ytest=np.load('/vol/work/mesa/wemb/Ytest'+str(emb)+'.npy')
		trainedmodel = load_model(dire+dirext+'flayerneurons.'+str(bestneuron)+'/weights.hdf5')
		Ypredict=trainedmodel.predict(Xtest, batch_size=32, verbose=0)
		[CFM,NCFM]=compute_cm(Ytest,Ypredict)
		sumrows=np.sum(CFM,axis=1)
		sumcols=np.sum(CFM,axis=0)
		P1=(CFM[0][0]/sumcols[0])*100
		P2=(CFM[1][1]/sumcols[1])*100
		P3=(CFM[2][2]/sumcols[2])*100

		R1=(CFM[0][0]/sumrows[0])*100
		R2=(CFM[1][1]/sumrows[1])*100
		R3=(CFM[2][2]/sumrows[2])*100
		f=open('/people/mesa/Desktop/results/NNearlystopping/confusionmatrix','a')
		f.write(str(emb)+' Context, best neurons '+str(bestneuron)+'\n \n')
		f.write(str(CFM))
		f.write('\n \n')
		f.write('Precision1 = '+str(P1)+'%\nPrecision2 = '+str(P2)+'%\nPrecision3 = '+str(P3)+'%\n\n')
		f.write('Rappel1 = '+str(R1)+'%\nRappel2 = '+str(R2)+'%\nRappel3 = '+str(R3))
		f.write('%\n'+'\n'+'\n'+'\n')
		f.close()

#plt.show()		
				
