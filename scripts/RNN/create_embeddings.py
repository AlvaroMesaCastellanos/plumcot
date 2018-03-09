#script that compute wembeddings and store them in a npdarray
#Arg1: seasons from and to compute embeddings
#Arg2: number of before/after words to
import sys
sys.path.append('/people/mesa/anacondaplumcot')

from os import listdir
import os
import json

import spacy   
import numpy as np        
import linecache
import time

nlp=spacy.load('en_core_web_lg')

numwords=sys.argv[1]
tipeoftrain=sys.argv[2]


if 'train' in tipeoftrain: #training seasons are 4 to 10
	seasonlist=list(range(4,11))
	saveword='train'
elif 'dev' in tipeoftrain: #dev seasons are 2 to 3
	seasonlist=list(range(2,4))
	saveword='dev'
elif 'tests' in tipeoftrain: #test season is 1
	seasonlist=list(range(1,2))
	saveword='test'

dire='/people/mesa/Desktop/BigBang/anotations/spacy.anotations/'

Y=[] #we will store classes here
classmatrix=np.eye(3, dtype=float) ##numpy array of classes
firstvector=1
for season in seasonlist:
	fulldire=dire+'tbbt.season'+str(season)
	for fich in listdir(fulldire):	
		if 'Episode' in fich: # if series we process it		
			f=open(fulldire+'/'+fich)
			print(fich)
			wordsarray=[]
			classarray=[]
			for line in f:
				lineas=line.split(' ')
				if ',' in lineas[0] or '.' in lineas[0] or '?' in lineas[0] or '!' in lineas[0]: #we dont take into account punctuation
					pass
				else:
					wordsarray.append(lineas[0])
					auxclass=lineas[4]
					auxclass=auxclass[:-1]
					classarray.append(auxclass)
			#we now process each labeled noun
			count=0
			lenarray=len(wordsarray)
			notnamebefore=1
			for item in classarray:
				if not 'X' in item and notnamebefore:##its a labeled word and the previous one wasnt a noun		
					notnamebefore=0				
					arr=list(range(count-int(numwords),count))+list(range(count+1,count+int(numwords)+1))
					arr = [x for x in arr if (x >= 0 and x<lenarray)] #we select values between boundaries
					#we now compute vectors
					flag=1
					for ind in arr:
						word=wordsarray[ind]
						wordprc=nlp(word)
						vecaux=wordprc.vector
						vecaux=vecaux.reshape(1,300)
						if flag:
							vec=vecaux
						else:
							vec=np.hstack((vec,vecaux))
						flag=0

					if vec.shape[1] < 300*int(numwords)*2: #we add zeros if this not match
						rest=2*300*int(numwords)-vec.shape[1]
						zeroarr=np.zeros((1,rest))
						vec=np.hstack((vec,zeroarr))
					##we store the final vector and class
					#we check the class
					if '1' in item:
						Yaux=(classmatrix[0][:])			
					elif '2' in item:
						Yaux=(classmatrix[1][:])
					elif '3' in item or not item: #if its 3 class or empty
						Yaux=(classmatrix[2][:])
					if firstvector:
						X=vec
						Y=Yaux
						firstvector=0			
					else:
						X=np.vstack((X,vec))
						Y=np.vstack((Y,Yaux))
				else:
					notnamebefore=1
				count=count+1
		
			f.close()
print(X.shape)
print(Y.shape)
np.save('/vol/work/mesa/wemb/X'+saveword+numwords,X)
np.save('/vol/work/mesa/wemb/Y'+saveword+numwords,Y)





