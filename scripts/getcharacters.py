from os import listdir
import os
import spacy   
import numpy as np        
import sys
import linecache
import time

dire='/home/meca/Desktop/BigBang/anotations/tbbt.season01.annotations.spacy/'

for fich in listdir(dire):
	f=open(dire+fich,'r')
	for line in f:
		lineasplit=line.split(' ')
		word=lineasplit[4]
		character=lineasplit[9]
		if '"' in character:
			pass
		else:
			print(word+'  '+character)
	f.close()
