##script that takes anotations and process 2 class this way
##our new classes will be 1st 2bef 2post 3rd
##2bef: the character appears in the previous line
##2post: the character appears in the next line

import sys

from os import listdir
import os
import json

import spacy   
import numpy as np        
import linecache
import time
from file_read_backwards import FileReadBackwards

dire='/people/mesa/Desktop/BigBang/anotations/spacy.anotations/'
diredest='/people/mesa/Desktop/BigBang/anotations/spacy.anotations.new/'


##we first create a single character list
for folder in listdir(dire):
	for files in listdir(dire+folder):
		f=open(dire+folder+'/'+files)
		singlecharacterlist=[]
		firstline=1
		for line in f:
			lines=line.split(' ')
			if firstline:
				firstline=0
				befcharacter=lines[1]
				singlecharacterlist.append(befcharacter)
			character=lines[1]
			if befcharacter != character: ##change in character we anotate it as befperson
				singlecharacterlist.append(character)


			befcharacter=lines[1]
		f.close()
		##with that list we anotate previous and following characters
		f=open(dire+folder+'/'+files)
		f2=open(diredest+folder+'/'+files,'a')
		firstline=1
		characterpointer=0
		for line in f:
			lines=line.split(' ')
			if firstline:
				firstline=0
				befcharacter=lines[1]
			character=lines[1]
			if befcharacter != character: ##change in character, we add 1 to the pointer		
				characterpointer=characterpointer+1
			
			if 'X' in line:
				f2.write(line)
			elif '1' in line:
				f2.write(line)
			elif '2' in line:
				if characterpointer-1>0:
					if lines[3] == singlecharacterlist[characterpointer-1]:
						f2.write(line[:-2]+'2\n')
				if characterpointer+1<len(singlecharacterlist)-1:
					if lines[3] == singlecharacterlist[characterpointer+1]:
						f2.write(line[:-2]+'3\n')
				
			elif '3' in line:
				f2.write(line[:-2]+'4\n')
			befcharacter=lines[1]
		f2.close()
		f.close()




