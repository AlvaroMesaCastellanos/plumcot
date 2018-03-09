from os import listdir
import os
import spacy                           # See "Installing spaCy"
import sys 

number=sys.argv[1]
dire='/home/meca/Desktop/BigBang/anotations/spacy.anotations/'
dire2='/home/meca/Desktop/BigBang/anotations/ano/'
for fich in listdir(dire+'tbbt.season'+number+'.annotations.spacy'):
	f=open(dire+'tbbt.season'+number+'.annotations.spacy/'+fich)
	f2=open(dire2+'tbbt.season'+number+'.annotations.spacy/'+fich,'a')	
	for line in f:
		lineas=line.split(' ')
		if 'person_name' in line:
			f2.write(lineas[4]+' '+lineas[7]+' '+lineas[8]+' '+lineas[9]+' '+lineas[10])		
		else:
			f2.write(lineas[4]+' '+lineas[7]+' X X X X\n')
	f.close()
	f2.close()

