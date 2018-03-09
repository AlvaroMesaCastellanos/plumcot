#script that evaluates 123 tagging performance over TBBT series according to rules seen in
#https://cvhci.anthropomatik.kit.edu/~mtapaswi/papers/WACV2016.pdf
from os import listdir
import os
import spacy                           # See "Installing spaCy"
import numpy as np
               # You are here.
import en_core_web_sm

nlp = en_core_web_sm.load()
#script used for an semi-automatic anotation of content, 1,2,3 person

dire='/people/mesa/Desktop/BigBang/anotations/spacy.anotations/'
dire2='/people/mesa/Desktop/BigBang/transcripts/'
seasonlist=['01','02','03','04','05','06','07','08','09','10']
labelsx=0
labels1=0
labels2=0
labels3=0
confusionmatrix=np.zeros((3, 4))
for season in seasonlist:
		#we first read the tagged file and then apply 123 rules and compute confusion matrix
	for fich in listdir(dire+'tbbt.season'+str(season)):	
		f=open(dire+'tbbt.season'+str(season)+'/'+fich)
		print(fich)	
		labelsgt=[]
		wordsgt=[]
		
		for line in f:
			line_s=line.split(' ')
			word=line_s[0]
			label=line_s[4]
			wordsgt.append(word)
			labelsgt.append(label[:-1])
		f.close()

		labels1=labels1+labelsgt.count('1')
		labels2=labels2+labelsgt.count('2')
		labels3=labels3+labelsgt.count('3')
		labelsx=labelsx+labelsgt.count('X')

		labels=[]
		f2=open(dire2+'Season'+str(season)+'/'+fich)
		cntlabel=0
		for line in f2:
			if 'Scene' in line: #we skip scene information
				pass
			else:
				
				line_s=line.split(':')
				character=line_s[0]
				if len(line_s)>1:
					sentence=line_s[1]
					sentence=sentence[1:-1]
					sentenceproc=nlp(sentence)
					longi=len(sentenceproc)
					#we now process each item in procsentence
					cntsentence=0
					for item in sentenceproc:
						if 'PERSON' in str(item.ent_type_):
							if 'am' == str(sentenceproc[cntsentence-1]) or "'m" == str(sentenceproc[cntsentence-1]): ##1st person classification
								if '1' in labelsgt[cntlabel]:
									confusionmatrix[0][0]=confusionmatrix[0][0]+1
								if '2' in labelsgt[cntlabel]:
									confusionmatrix[1][0]=confusionmatrix[1][0]+1
								if '3' in labelsgt[cntlabel]:
									confusionmatrix[2][0]=confusionmatrix[2][0]+1
					
							elif 'are' == str(sentenceproc[cntsentence-1]) or "'re" == str(sentenceproc[cntsentence-1]): ##2nd person classification							
								if '1' in labelsgt[cntlabel]:
									confusionmatrix[0][1]=confusionmatrix[0][1]+1
								if '2' in labelsgt[cntlabel]:
									confusionmatrix[1][1]=confusionmatrix[1][1]+1
								if '3' in labelsgt[cntlabel]:
									confusionmatrix[2][1]=confusionmatrix[2][1]+1
						
							elif 'IN' in (sentenceproc[cntsentence-1].tag_) or 'CC' in (sentenceproc[cntsentence-1].tag_): #3rd person, prep plus name								
								if '1' in labelsgt[cntlabel]:
									confusionmatrix[0][2]=confusionmatrix[0][2]+1
								if '2' in labelsgt[cntlabel]:
									confusionmatrix[1][2]=confusionmatrix[1][2]+1
								if '3' in labelsgt[cntlabel]:
									confusionmatrix[2][2]=confusionmatrix[2][2]+1
								
							elif cntsentence<longi-1 and 'VBZ' in (sentenceproc[cntsentence+1].tag_): #3rd person, prep plus name:
								if '1' in labelsgt[cntlabel]:
									confusionmatrix[0][2]=confusionmatrix[0][2]+1
								if '2' in labelsgt[cntlabel]:
									confusionmatrix[1][2]=confusionmatrix[1][2]+1
								if '3' in labelsgt[cntlabel]:
									confusionmatrix[2][2]=confusionmatrix[2][2]+1

							elif cntsentence<longi-1 and str(sentenceproc[cntsentence+1])==',': #2nd person, name plus comma:
								if '1' in labelsgt[cntlabel]:
									confusionmatrix[0][1]=confusionmatrix[0][1]+1
								if '2' in labelsgt[cntlabel]:
									confusionmatrix[1][1]=confusionmatrix[1][1]+1
								if '3' in labelsgt[cntlabel]:
									confusionmatrix[2][1]=confusionmatrix[2][1]+1
							
							else: #we didnt found a label
								if '1' in labelsgt[cntlabel]:
									confusionmatrix[0][3]=confusionmatrix[0][3]+1
								if '2' in labelsgt[cntlabel]:
									confusionmatrix[1][3]=confusionmatrix[1][3]+1
								if '3' in labelsgt[cntlabel]:
									confusionmatrix[2][3]=confusionmatrix[2][3]+1

						cntlabel=cntlabel+1
						cntsentence=cntsentence+1
		f2.close()
		#print(len(labels))
		#print(len(labelsgt))


print(confusionmatrix)

print('#X labels = '+str(labelsx))
print('#1 labels = '+str(labels1))
print('#2 labels = '+str(labels2))
print('#3 labels = '+str(labels3))

sumrows=np.sum(confusionmatrix,axis=1)
sumcols=np.sum(confusionmatrix,axis=0)
p1=(confusionmatrix[0][0]/sumcols[0])*100
p2=(confusionmatrix[1][1]/sumcols[1])*100
p3=(confusionmatrix[2][2]/sumcols[2])*100

r1=(confusionmatrix[0][0]/sumrows[0])*100
r2=(confusionmatrix[1][1]/sumrows[1])*100
r3=(confusionmatrix[2][2]/sumrows[2])*100
print('P1 = '+str(p1))
print('P2 = '+str(p2))
print('P3 = '+str(p3))

print('R1 = '+str(r1))
print('R2 = '+str(r2))
print('R3 = '+str(r3))

















