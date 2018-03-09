#preprocessing of anotations, putting it into spacy way

from os import listdir
import os
import spacy                           # See "Installing spaCy"
nlp = spacy.load('en')                 # You are here.

dire='/people/mesa/Desktop/BigBang/anotations/'



for fich in listdir(dire+'tbbt.season01.annotations'):
	f=open(dire+'tbbt.season01.annotations/'+fich)
	f2=open(dire+'tbbt.season01.annotations.spacy/'+fich,'a')
	for line in f:
		splitted=line.split(' ')
		wordproc=nlp(splitted[4])
		for item in wordproc:
			if "'s" in str(item) and 'person_name' in line: #'s isnt name in spacy
				splitted[4]=str(item)
				sentence=' '.join(splitted[0:8])
				f2.write(sentence+' " " "\n')
			else:
				splitted[4]=str(item)
				sentence=' '.join(splitted)
				f2.write(sentence)
	f.close()
