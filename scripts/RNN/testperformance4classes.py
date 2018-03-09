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


nlp = spacy.load('en')   ## load the lm

##we load character dictionary

with open('/people/mesa/Desktop/scripts/chclist', 'r') as content_file:
    content = content_file.read()
DICT = json.loads(content)

###############################

NNmodels='/vol/work/mesa/NNearlystopping4classes/'
LSTMmodels='/vol/work/mesa/LSTMearlystopping4classes/'

testdire='/people/mesa/Desktop/BigBang/transcripts/Season1/'

def computefscore(P,R,beta):
	if P*R==0:
		return 0
	else:
		return ((1+pow(beta,2))*((P*R)/((pow(beta,2)*P)+R)))

modelslist=[NNmodels,LSTMmodels]
for modeltype in modelslist: #we evaluate results on NN and LSTM models
	if 'LSTM' in modeltype:
		f=open('/vol/work/mesa/RESULTS/resultsLSTMearlystoppingnoweights4classes','a')
	elif 'NN' in modeltype:
		f=open('/vol/work/mesa/RESULTS/resultsNNearlystoppingnoweights4classes','a')

	for nembedings in listdir(modeltype):
		contextaux=nembedings.split('.')
		context=int(contextaux[0])
		if 'embeding' in nembedings:	
			for nneurons in listdir(modeltype+nembedings):
				layeraux=nneurons.split('.')
				layeraux=layeraux[1]
				trainedmodel = load_model(modeltype+nembedings+'/'+nneurons+'/weights.hdf5')
				##once we load the trained model we find the Named entities and classify it as 1,2,3 person
				##if we classify it as 1 person we can name the current line
				##if we classify it as 2 person we can name the previous and next line
				##if we classify it as 3 person we do not do anything
				totallines=0

				totalacc1=0
				totalrapp1=0
				totalprec1=0

				totalacc2bef=0
				totalrapp2bef=0
				totalprec2bef=0

				totalacc2aft=0
				totalrapp2aft=0
				totalprec2aft=0

				totalacc=0
				totalrapp=0
				totalprec=0

				pacc1=0
				pacc2bef=0
				pacc2aft=0
				pacc=0

				for testfile in listdir(testdire):
					ftest=open(testdire+testfile,'r')
					classed_name1=[]
					classed_name2bef=[]
					classed_name2aft=[]

					scores1=[]
					scores2bef=[]
					scores2aft=[]

					lines_names=[]
					nlines=-1
					for line in ftest:
						if 'Scene' in line: ##we pass scene information
							pass
						else:
							nlines=nlines+1
							classed_name1.append('0') ##initialization
							classed_name2aft.append('0')
							classed_name2bef.append('0')

							scores1.append('0')
							scores2bef.append('0')
							scores2aft.append('0')

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
							firstname2before=1 #flag to indicate a second class name in a line
							firstname2after=1							
			
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
										#f.write(name)
										endidx=idx-1
										#f.write(str(startidx)+' '+str(endidx))
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
											Ypredict=trainedmodel.predict(vec, batch_size=32, verbose=0)
											predicted_class=np.argmax(Ypredict,axis=-1)
											##
											if predicted_class==3: #3rd person
												pass
											if predicted_class==2: #2nd person after
												if firstname2after:
													classed_name2aft[nlines]=norm_name
													score2aft=Ypredict[0][1]
													scores2aft[nlines]=score2aft
													firstname2after=0
												else:
													if Ypredict[0][1]>score2aft:
														score2aft=Ypredict[0][1]
														scores2aft[nlines]=score2aft
														classed_name2aft[nlines]=norm_name
											if predicted_class==1: #2nd person before
												if firstname2before:
													classed_name2bef[nlines]=norm_name
													score2bef=Ypredict[0][1]
													scores2bef[nlines]=score2bef
													firstname2before=0
												else:
													if Ypredict[0][1]>score2bef:
														score2bef=Ypredict[0][1]
														scores2bef[nlines]=score2bef
														classed_name2bef[nlines]=norm_name
											if predicted_class==0: #1st person
												if firstname1:
													classed_name1[nlines]=norm_name
													score1=Ypredict[0][0]
													scores1[nlines]=score1
													firstname1=0
												else:
													if Ypredict[0][0]>score1:
														score1=Ypredict[0][0]
														scores1[nlines]=score1
														classed_name1[nlines]=norm_name

												
									
									
								idx=idx+1

					#we preprocess class2
					
			
					for cnt in range(0,len(classed_name2bef)-1):
						if '0' not in str(classed_name2bef[cnt]):
							if cnt > 0:
								scores2bef[cnt-1]=scores2bef[cnt]
								scores2bef[cnt]='0'
								classed_name2bef[cnt-1]=classed_name2bef[cnt]
								classed_name2bef[cnt]='0'
						if '0' not in str(classed_name2aft[cnt]):
							if cnt < len(classed_name2aft)-1:
								scores2aft[cnt+1]=scores2aft[cnt]
								scores2aft[cnt]='0'
								classed_name2aft[cnt+1]=classed_name2aft[cnt]
								classed_name2aft[cnt]='0'
					
					##we make the final list
					final_names=['0']*len(classed_name1)
					for cnt in range(0,len(lines_names)-1):
						if float(scores1[cnt]) > float(scores2bef[cnt]) and float(scores1[cnt]) > float(scores2aft[cnt]):
							final_names[cnt]=classed_name1[cnt]
						elif float(scores2bef[cnt]) > float(scores1[cnt]) and float(scores2bef[cnt]) > float(scores2aft[cnt]):
							final_names[cnt]=classed_name2bef[cnt]
						elif float(scores2aft[cnt]) > float(scores1[cnt]) and float(scores2aft[cnt]) > float(scores2bef[cnt]):
							final_names[cnt]=classed_name2aft[cnt]
					##now we compare the classed names

					for cnt in range(0,len(lines_names)-1): ##if the name match we add 1
						if lines_names[cnt] == classed_name1[cnt]:
							pacc1=pacc1+1
						if lines_names[cnt] == classed_name2bef[cnt]:
							pacc2bef=pacc2bef+1
						if lines_names[cnt] == classed_name2aft[cnt]:
							pacc2aft=pacc2aft+1
						if lines_names[cnt] == final_names[cnt]:
							pacc=pacc+1
				
					totalprec1=totalprec1+(len(classed_name1)-classed_name1.count('0'))
					
					totalrapp1=totalrapp1+(len(classed_name1)-classed_name1.count('0'))
					
					totalprec2bef=totalprec2bef+(len(classed_name2bef)-classed_name2bef.count('0'))
					
					totalrapp2bef=totalrapp2bef+(len(classed_name2bef)-classed_name2bef.count('0'))

					totalprec2aft=totalprec2aft+(len(classed_name2aft)-classed_name2aft.count('0'))
					
					totalrapp2aft=totalrapp2aft+(len(classed_name2aft)-classed_name2aft.count('0'))
					
					totalprec=totalprec+(len(final_names)-final_names.count('0'))
					
					totalrapp=totalrapp+(len(final_names)-final_names.count('0'))

					totallines=totallines+len(lines_names)



				if int(totalprec1) == 0:
					precision1=0
				else:
					precision1=int(pacc1)/int(totalprec1)

				if int(totalprec2bef) == 0:
					precision2bef=0
				else:
					precision2bef=int(pacc2bef)/int(totalprec2bef)

				if int(totalprec2aft) == 0:
					precision2aft=0
				else:
					precision2aft=int(pacc2aft)/int(totalprec2aft)

				if int(totalprec) == 0:

					precision=0
				else:
					precision=int(pacc)/int(totalprec)

				rapel1=int(totalrapp1)/int(totallines)
				rapel2bef=int(totalrapp2bef)/int(totallines)
				rapel2aft=int(totalrapp2aft)/int(totallines)
				rapel=int(totalrapp)/int(totallines)

				fscore1a=computefscore(precision1,rapel1,0.1)
				fscore1b=computefscore(precision1,rapel1,1)
				fscore1c=computefscore(precision1,rapel1,10)

				fscore2abef=computefscore(precision2bef,rapel2bef,0.1)
				fscore2bbef=computefscore(precision2bef,rapel2bef,1)
				fscore2cbef=computefscore(precision2bef,rapel2bef,10)
				
				fscore2aaft=computefscore(precision2aft,rapel2aft,0.1)
				fscore2baft=computefscore(precision2aft,rapel2aft,1)
				fscore2caft=computefscore(precision2aft,rapel2aft,10)

				fscorea=computefscore(precision,rapel,0.1)
				fscoreb=computefscore(precision,rapel,1)
				fscorec=computefscore(precision,rapel,10)



				f.write('1 '+str(contextaux[0])+' '+str(layeraux)+' '+str(pacc1)+' '+str(precision1)+' '+str(rapel1)+' '+str(fscore1a)+' '+str(fscore1b)+' '+str(fscore1c)+' '+str(pacc)+' '+str(totallines)+' '+str(int(pacc)/int(totallines))+' '+str(int(pacc)/int(totalprec))+'\n')
				f.write('21 '+str(contextaux[0])+' '+str(layeraux)+' '+str(pacc2bef)+' '+str(precision2bef)+' '+str(rapel2bef)+' '+str(fscore2abef)+' '+str(fscore2bbef)+' '+str(fscore2cbef)+' '+str(pacc)+' '+str(totallines)+' '+str(int(pacc)/int(totallines))+' '+str(int(pacc)/int(totalprec))+'\n')
				f.write('22 '+str(contextaux[0])+' '+str(layeraux)+' '+str(pacc2aft)+' '+str(precision2aft)+' '+str(rapel2aft)+' '+str(fscore2aaft)+' '+str(fscore2baft)+' '+str(fscore2caft)+' '+str(pacc)+' '+str(totallines)+' '+str(int(pacc)/int(totallines))+' '+str(int(pacc)/int(totalprec))+'\n')	
				f.write('12 '+str(contextaux[0])+' '+str(layeraux)+' '+str(pacc)+' '+str(precision)+' '+str(rapel)+' '+str(fscorea)+' '+str(fscoreb)+' '+str(fscorec)+' '+str(pacc)+' '+str(totallines)+' '+str(int(pacc)/int(totallines))+' '+str(int(pacc)/int(totalprec))+'\n')	
				
	f.close()
