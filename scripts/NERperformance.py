from os import listdir
import os
import spacy   
import numpy as np        
import sys
import linecache
import time
#Script that takes transcripts and check it from ground thruth data

nlp = spacy.load('en')
dire='/home/meca/Desktop/BigBang/anotations/spacy.anotations/tbbt.season01/'
dire2='/home/meca/Desktop/BigBang/transcripts/Season01/'

confusionmatrix=np.zeros((2, 2))
for fich in listdir(dire2):
	if '~' in fich:
		pass
	else:
		wrdsgt=[]
		wrdst=[]
		wordscheck=[]
		wordsf=[]
		f=open(dire2+fich)
		f2=open(dire+fich)
		for line in f2:#we append person name with 1 and else with 0
			lineasplit=line.split(' ')
			wrdsgt.append(lineasplit[4])			
			if 'person_name' in line:
				wordsf.append('T')
			else:
				wordsf.append('F')
		for line in f:
			line_s=line.split(':')
			character=line_s[0]
			if len(line_s)>1:
				sentence=line_s[1]
				sentence=sentence[1:-1]
				if 'Scene' in character: ##we pass scene information
					pass
				else:
					sentenceproc=nlp(sentence)
					for item in sentenceproc:
						wrdst.append(str(item))
						if 'PERSON' in str(item.ent_type_): #if person match 1 else 0
							wordscheck.append('T')
						else:
							wordscheck.append('F')

		f.close()
		f2.close()
		for i in range(0,len(wordsf)):
			if 'T' in str(wordsf[i]) and 'T' in str(wordscheck[i]):
				confusionmatrix[0][0]=confusionmatrix[0][0]+1
			elif 'T' in str(wordsf[i]) and 'F' in str(wordscheck[i]):
				confusionmatrix[0][1]=confusionmatrix[0][1]+1
				#input(str(wrdsgt[i]+' '+wrdst[i])+dire+fich+'.speakername.ctm   '+str(i))
			elif 'F' in str(wordsf[i]) and 'T' in str(wordscheck[i]):
				confusionmatrix[1][0]=confusionmatrix[1][0]+1
				#input(str(wrdsgt[i]+' '+wrdst[i])+dire+fich+'.speakername.ctm   '+str(i))
			else:
				confusionmatrix[1][1]=confusionmatrix[1][1]+1
		
RECALL=confusionmatrix[0][0]/(confusionmatrix[0][0]+confusionmatrix[0][1])
PRECISION=confusionmatrix[0][0]/(confusionmatrix[0][0]+confusionmatrix[1][0])
print('RECALL: '+str(RECALL*100)+' %')
print('PRECISION: '+str(PRECISION*100)+' %')
print(confusionmatrix)


