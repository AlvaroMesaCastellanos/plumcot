from os import listdir
import numpy
import os
import spacy                           # See "Installing spaCy"
nlp = spacy.load('en')                 # You are here.

#script used for NER anotation

dire='/people/mesa/Desktop/BigBang/anotations/spacy.anotations/tbbt.season01/'
confusionmatrix=numpy.zeros((2, 2))

for fich in listdir(dire):
	f=open(dire+fich)
	for line in f:
		lineproc=nlp(line)
		word=lineproc[0]
		if 'person_name' in line:
			if 'PERSON' in str(word.ent_type_):
				confusionmatrix[0][0]=confusionmatrix[0][0]+1
			else:
				confusionmatrix[0][1]=confusionmatrix[0][1]+1
		else:
			if 'PERSON' in str(word.ent_type_):
				confusionmatrix[1][0]=confusionmatrix[1][0]+1
			else:
				confusionmatrix[1][1]=confusionmatrix[1][1]+1
	f.close()
			
RECALL=confusionmatrix[0][0]/(confusionmatrix[0][0]+confusionmatrix[1][0])
PRECISION=confusionmatrix[0][0]/(confusionmatrix[0][0]+confusionmatrix[0][1])
print('RECALL: '+str(RECALL*100)+' %')
print('PRECISION: '+str(PRECISION*100)+' %')
print(confusionmatrix)

