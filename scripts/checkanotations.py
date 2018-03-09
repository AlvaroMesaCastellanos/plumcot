from os import listdir
import os
import spacy   
import numpy       
import sys
import linecache
import time
#Script that takes transcripts and anotates it from ground thruth data
                # You are here.
nlp = spacy.load('en')
seas=sys.argv[1]
dire='/home/meca/Desktop/BigBang/anotations/spacy.anotations/tbbt.season'+seas+'/'
dire2='/home/meca/Desktop/BigBang/transcripts/Season'+seas+'/'


for fich in listdir(dire2):
	wordscheck=[]
	wordsf=[]
	if '~' in fich:
		pass
	else:
		count=0
		f=open(dire2+fich)
		f2=open(dire+fich)
		for line in f2:#we append words to check
			linea=line.split(' ')
			wordscheck.append(linea[0])
		longf2=len(wordscheck)
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
						count=count+1
						wordsf.append(str(item))

		f.close()
		f2.close()
		longf=len(wordsf)
		minlong=min(longf,longf2)
		#now we check the differences between two lists
		for i in range(0,minlong-1):
			if str(wordsf[i]) == str(wordscheck[i]):
				pass
				#print('OK'+str(i))
				#print(str(wordsf[i])+' '+str(wordscheck[i]))
			else:
				print(str(i))
				print(str(wordsf[i])+' '+str(wordscheck[i]))
				print(fich)
				input()
	if longf!=longf2:
		print(fich)
		print(str(longf)+'  '+str(longf2))

