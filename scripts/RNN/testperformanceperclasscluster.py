from os import listdir
import os
import json
import spacy
import numpy as np   
import math     
import sys
import linecache
import time
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.models import load_model
from sklearn.metrics import confusion_matrix


nlp = spacy.load('en')   ## load the lm

##we load character and cluster dictionary

with open('/people/mesa/Desktop/scripts/chclist', 'r') as content_file:
    content = content_file.read()
DICT = json.loads(content)

with open('/people/mesa/Desktop/scripts/clustersS1', 'r') as content_file2:
    content = content_file2.read()
clusterdict = json.loads(content)

###############################

NNmodels='/vol/work/mesa/NNearlystoppingperclass/'
LSTMmodels='/vol/work/mesa/LSTMearlystoppingperclass/'

testdire='/people/mesa/Desktop/BigBang/transcripts/Season1/'

def computefscore(P,R,beta):
	if P*R==0:
		return 0
	else:
		return ((1+pow(beta,2))*((P*R)/((pow(beta,2)*P)+R)))


modelslist=[NNmodels,LSTMmodels]
nneuronslist=list(range(1,11))
nneuronslist=[str(int(math.pow(2,n))) for n in nneuronslist]

for modeltype in modelslist: #we evaluate results on NN and LSTM models
	if 'LSTM' in modeltype:
		f=open('/vol/work/mesa/RESULTS/resultsLSTMearlystoppingperclasscluster','a')
	elif 'NN' in modeltype:
		f=open('/vol/work/mesa/RESULTS/resultsNNearlystoppingperclasscluster','a')

	for nembedings in listdir(modeltype):
		contextaux=nembedings.split('.')
		context=int(contextaux[0])
		if 'embeding' in nembedings:	
			for nneurons in nneuronslist:

				trainedmodel1 = load_model(modeltype+nembedings+'/1flayerneurons.'+nneurons+'/weights.hdf5')
				trainedmodel2 = load_model(modeltype+nembedings+'/2flayerneurons.'+nneurons+'/weights.hdf5')
				trainedmodel3 = load_model(modeltype+nembedings+'/3flayerneurons.'+nneurons+'/weights.hdf5')
				##once we load the 3 trained models per class we find the Named entities and classify it as 1,2,3 person
				##if we classify it as 1 person we can name the current line
				##if we classify it as 2 person we can name the previous and next line
				##if we classify it as 3 person we do not do anything
				totallines=0

				totalacc1=0
				totalrapp1=0
				totalprec1=0

				totalacc2=0
				totalrapp2=0
				totalprec2=0

				totalacc=0
				totalrapp=0
				totalprec=0

				pacc1=0
				pacc2=0
				pacc=0

				for testfile in listdir(testdire):
					ftest=open(testdire+testfile,'r')
					classed_name1=[]
					classed_name2=[]
					scores1=[]
					scores2=[]
					lines_names=[]
					nlines=-1
					for line in ftest:
						if 'Scene' in line: ##we pass scene information
							pass
						else:
							nlines=nlines+1
							classed_name1.append('0') ##initialization
							classed_name2.append('0')
							scores1.append('0')
							scores2.append('0')
							lines=line.split(':')
							character=lines[0]
							dialog=lines[1]
							lineproc=nlp(dialog[1:-1])
							classarray=[]
							namesarray=[]
							lines_names.append(DICT.get(character,'None'))
							for token in lineproc:

								if 'PERSON' in (str(token.ent_type_)): ##we found an Entity
									classarray.append('1')
									namesarray.append(str(token))
								else:
									classarray.append('0')
									namesarray.append(str(token))

							##we look now for the names and classify them
							
							firstname1=1 #flag to indicate a first class name in a line
							firstname2=1 #flag to indicate a second class name in a line
					
							idx=0
							namebefore=0
							name=[]

							for item in classarray:
								if '1' in item:
									if not namebefore:
										startidx=idx
									name.append(namesarray[idx])
									namebefore=1
								else:
									if namebefore:
										namebefore=0
										#print(name)
										endidx=idx-1
										#print(str(startidx)+' '+str(endidx))
										norm_name=DICT.get(' '.join(name),'None')
										name=[]
										if 'None' in norm_name: #we didnt find it
											pass
										else: #we process it
											wordrange=list(range(startidx-context,startidx))+list(range(endidx+1,endidx+1+context))
											wordrange = [x for x in wordrange if (x >= 0 and x<len(classarray))]
											#we now compute vectors for each name
											flag=1
											for ind in wordrange:
												word=namesarray[ind]
												wordprc=nlp(word)
												vecaux=wordprc.vector
												vecaux=vecaux[0:300] ##300 size vectors
												vecaux=vecaux.reshape(1,300)
												if flag:
													vec=vecaux
													flag=0
												else:
													vec=np.hstack((vec,vecaux))

											if vec.shape[1] < 300*context*2: #we add zeros if this not match
												rest=2*300*context-vec.shape[1]
												zeroarr=np.zeros((1,rest))
												vec=np.hstack((vec,zeroarr))
											#we now predict the class
											if 'LSTM' in modeltype:
												dim=vec.shape
												vec=vec.reshape(dim[0],2*int(context),300)
											else:
												pass
											scoresperclass=np.zeros((1,3)) ##we will store the 3 scores in this array
											scoresperclass[0][0]=trainedmodel1.predict(vec, batch_size=32, verbose=0)
											scoresperclass[0][1]=trainedmodel2.predict(vec, batch_size=32, verbose=0)
											scoresperclass[0][2]=trainedmodel3.predict(vec, batch_size=32, verbose=0)

											predicted_class=np.argmax(scoresperclass,axis=-1)
											##
											if predicted_class==2: #3rd person
												pass
											if predicted_class==1: #2nd person
												if firstname2:
													classed_name2[nlines]=clusterdict.get(norm_name,'None') ##we assign the cluster
													score2=scoresperclass[0][1]
													scores2[nlines]=score2
													firstname2=0
												else:
													if scoresperclass[0][1]>score2:
														score2=scoresperclass[0][1]
														scores2[nlines]=score2
														classed_name2[nlines]=clusterdict.get(norm_name,'None') ##we assign the cluster
											if predicted_class==0: #1st person
												if firstname1:
													classed_name1[nlines]=clusterdict.get(norm_name,'None') ##we assign the cluster
													score1=scoresperclass[0][0]
													scores1[nlines]=score1
													firstname1=0
												else:
													if scoresperclass[0][0]>score1:
														score1=scoresperclass[0][0]
														scores1[nlines]=score1
														classed_name1[nlines]=clusterdict.get(norm_name,'None') ##we assign the cluster

												
									
									
								idx=idx+1
					
					#we preprocess class2
					class2=['0']*len(classed_name2)
			
					for cnt in range(0,len(classed_name2)-1):
						if '0' not in str(classed_name2[cnt]):
							if cnt > 0 and cnt < len(classed_name2)-1:
								scores2[cnt-1]=scores2[cnt]
								scores2[cnt+1]=scores2[cnt]
								scores2[cnt]='0'
								class2[cnt-1]=classed_name2[cnt]
								class2[cnt+1]=classed_name2[cnt]
								class2[cnt]='0'
					##we make the final list
					final_names=['0']*len(classed_name2)
					for cnt in range(0,len(lines_names)-1):
						if float(scores1[cnt])>float(scores2[cnt]):
							final_names[cnt]=classed_name1[cnt]
						else:
							final_names[cnt]=class2[cnt]

					##in this step we add CLUSTERING INFORMATION
					index=0
					for element in lines_names:
						if element in classed_name1: ##if we retrieve the cluster we assign it
							classed_name1[index]=element
						elif element in class2:
							class2[index]=element
						elif element in final_names:
							final_names[index]=element
						index=index+1

					##now we compare the classed names

					for cnt in range(0,len(lines_names)-1): ##if the name match we add 1
						if lines_names[cnt] == classed_name1[cnt]:
							pacc1=pacc1+1
						if lines_names[cnt] == class2[cnt]:
							pacc2=pacc2+1
						if lines_names[cnt] == final_names[cnt]:
							pacc=pacc+1
				
					totalprec1=totalprec1+(len(classed_name1)-classed_name1.count('0'))
					
					totalrapp1=totalrapp1+(len(classed_name1)-classed_name1.count('0'))
					
					totalprec2=totalprec2+(len(class2)-class2.count('0'))
					
					totalrapp2=totalrapp2+(len(class2)-class2.count('0'))
					
					totalprec=totalprec+(len(final_names)-final_names.count('0'))
					
					totalrapp=totalrapp+(len(final_names)-final_names.count('0'))

					totallines=totallines+len(lines_names)


				if int(totalprec1) == 0:
					precision1=0
				else:
					precision1=int(pacc1)/int(totalprec1)

				if int(totalprec2) == 0:
					precision2=0
				else:
					precision2=int(pacc2)/int(totalprec2)
				if int(totalprec) == 0:

					precision=0
				else:
					precision=int(pacc)/int(totalprec)

				rapel1=int(totalrapp1)/int(totallines)
				rapel2=int(totalrapp2)/int(totallines)
				rapel=int(totalrapp)/int(totallines)

				fscore1a=computefscore(precision1,rapel1,0.1)
				fscore1b=computefscore(precision1,rapel1,1)
				fscore1c=computefscore(precision1,rapel1,10)

				fscore2a=computefscore(precision2,rapel2,0.1)
				fscore2b=computefscore(precision2,rapel2,1)
				fscore2c=computefscore(precision2,rapel2,10)

				fscorea=computefscore(precision,rapel,0.1)
				fscoreb=computefscore(precision,rapel,1)
				fscorec=computefscore(precision,rapel,10)



				f.write('1 '+str(contextaux[0])+' '+str(nneurons)+' '+str(pacc1)+' '+str(precision1)+' '+str(rapel1)+' '+str(fscore1a)+' '+str(fscore1b)+' '+str(fscore1c)+' '+str(pacc)+' '+str(totallines)+' '+str(int(pacc)/int(totallines))+' '+str(int(pacc)/int(totalprec))+'\n')
				f.write('2 '+str(contextaux[0])+' '+str(nneurons)+' '+str(pacc2)+' '+str(precision2)+' '+str(rapel2)+' '+str(fscore2a)+' '+str(fscore2b)+' '+str(fscore2c)+' '+str(pacc)+' '+str(totallines)+' '+str(int(pacc)/int(totallines))+' '+str(int(pacc)/int(totalprec))+'\n')	
				f.write('12 '+str(contextaux[0])+' '+str(nneurons)+' '+str(pacc)+' '+str(precision)+' '+str(rapel)+' '+str(fscorea)+' '+str(fscoreb)+' '+str(fscorec)+' '+str(pacc)+' '+str(totallines)+' '+str(int(pacc)/int(totallines))+' '+str(int(pacc)/int(totalprec))+'\n')	
				
	f.close()




