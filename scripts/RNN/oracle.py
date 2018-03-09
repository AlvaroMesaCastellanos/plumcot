from os import listdir
import os
import json
import spacy
import numpy as np        
import sys
import linecache
import time
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

dire='/people/mesa/Desktop/BigBang/anotations/spacy.anotations/tbbt.season1/'

linesanotated=0 #lines we will anotate correctly
totallines=0 #total lines of the transcript


for fich in listdir(dire):
	f=open(dire+fich,'r')
	firstline=1
	class1list=[]
	class2list=[]

	labelslist=[]
	for line in f:
		lines=line.split(' ')
		labelslist.append(lines[4][:-1])
		if firstline:
			characterbef=lines[1]
			character=characterbef
			firstline=0
		else:
			character=lines[1]
			if character not in characterbef: #change of speaker we analize how many 1 labels and 2 labels there were in the list
				if '1' in labelslist:
					class1list.append('1')
				else:
					class1list.append('0')
				if '2' in labelslist:
					class2list.append('1')
				else:
					class2list.append('0')
	
				labelslist=[]
			characterbef=character

	##now, with information if class1/2 appeared in the line we can extract the final accuracy
	#we preprocess class2
	class2=['0']*len(class2list)

	for cnt in range(0,len(class2list)-1):
		if '0' not in str(class2list[cnt]):
			if cnt > 0 and cnt < len(class2list)-1:
				class2[cnt-1]=class2list[cnt]
				class2[cnt+1]=class2list[cnt]
				class2[cnt]='0'

	#now we process final data
	
	for cnt in range(0,len(class2list)-1): ##if 
		if '1' in class1list[cnt]:
			linesanotated=linesanotated+1
		elif '1' in class2[cnt]:
			linesanotated=linesanotated+1
		else:
			pass


	totallines=totallines+len(class2)
	##once we have all the lines we start to process the information
	f.close()

print('Oracle Accuracy: '+str((int(linesanotated)/int(totallines))*100)+'%')
