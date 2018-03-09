from os import listdir
import os
import spacy                           # See "Installing spaCy"
nlp = spacy.load('en')                 # You are here.

#script used for an semi-automatic anotation of content, 1,2,3 person

dire='/home/meca/Desktop/BigBang/transcripts/Season1/'
dire2='/home/meca/Desktop/BigBang/anotations/tbbt.season01.annotations.123/'

for fich in listdir(dire):
	if '~' in fich:
		pass
	else:
		f=open(dire+fich)
		f2=open(dire2+fich,'a')
		for line in f:
			line_s=line.split(':')
			character=line_s[0]
			if len(line_s)>1:
				sentence=line_s[1]
				sentence=sentence[1:-1]
				sentenceproc=nlp(sentence)
				longi=len(sentenceproc)
				i=0
				for item in sentenceproc:
					M=0
			    #########R1########
					if i>1:
						if ('be' in sentenceproc[i-1].lemma_) and ('VBP' in sentenceproc[i-1].tag_) and ('PERSON' in str(sentenceproc[i].ent_type_)) and M==0:
							if 'I' in str(sentenceproc[i-2]):                   
								f2.write(str(sentenceproc[i])+'----1st'+'\n')
								M=1
							else:
								f2.write(str(sentenceproc[i])+'----2nd'+'\n')
								M=1   
				    ##########R2#######
					if i>0:
						if (',' in str(sentenceproc[i-1])) and 'PERSON' in str(sentenceproc[i].ent_type_) and M==0:# NAME AFTER COMMA:
							f2.write(str(sentenceproc[i])+'----2nd'+'\n')
							M=1
				    ##########R4#######
						elif ('IN' in sentenceproc[i-1].tag_) and ('PERSON' in str(sentenceproc[i].ent_type_)) and M==0:#R4 PREP PLUS NOUN
							f2.write(str(sentenceproc[i])+'----3rd'+'\n')
							M=1
					if i<longi-1:
						if (',' in str(sentenceproc[i+1])) and M==0 and 'PERSON' in str(sentenceproc[i].ent_type_):# NAME BEFORE COMMA:
							f2.write(str(sentenceproc[i])+'----2nd'+'\n')
							M=1
						elif ('?' in str(sentenceproc[i+1])) and M==0 and 'PERSON' in str(sentenceproc[i].ent_type_):#NOUNandQUESTION+R2-:
							f2.write(str(sentenceproc[i])+'----2nd'+'\n')
							M=1
				    ##########R3#######
						elif (i<longi-1) and ('VBZ' in sentenceproc[i+1].tag_) and M==0:#R3 VERB IN 3 PERSON SINGULAR
							if (str(sentenceproc[i]).startswith('he')) or (str(sentenceproc[i]).startswith('she')):
								f2.write(str(sentenceproc[i])+'----3rd'+'\n')
								M=1
					if (i<longi-2):
						if ('VBZ' in sentenceproc[i+2].tag_) and M==0:#R3 VERB IN 3 PERSON SINGULAR
							if (str(sentenceproc[i]).startswith('he')) or (str(sentenceproc[i]).startswith('she')):
								f2.write(str(sentenceproc[i])+'----3rd'+'\n')
								M=1

				    ##########R5#######
					if (i>1):
						if ('CC' in sentenceproc[i-1].tag_) and M==0:##enumeration
							if ('PERSON' in str(sentenceproc[i-2].ent_type_)) and ('PERSON' in str(sentenceproc[i].ent_type_)):
								f2.write(str(sentenceproc[i])+'----3rd'+'\n')
								M=1
					if M==0:
						f2.write(str(sentenceproc[i])+'\n')
					i=i+1 

