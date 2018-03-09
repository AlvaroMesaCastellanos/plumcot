#Script that performs NamedEntity via spaCy and anotate raw text, following this format
#TheBigBangTheory.SeasonXX.EpisodeXX.en Alright(word) sheldon_cooper(character) " " " , e.g
#TheBigBangTheory.Season01.Episode05.en Leonard lesley_winkle person_name leonard_hofstadter addressee
#third " can be either speaker(first person),addressee(second person) or object(third person)

from os import listdir
import os
import json
import spacy   
import numpy as np        
import sys
import linecache
import time
import colorama
from colorama import Fore, Back, Style


def createdict(fpont): #function that creates a dictionary, receives a file pointer, fpont
	dictionary={}	
	for line in fpont:
		lines=line.split(',')
		names=lines[0]
		key=lines[1]
		for item in names.split(';'):
			dictionary[item]=key[:-1]

	with open('chclist','a') as f2:
		json.dump(dictionary, f2)
		f2.close() 
		
def lookupdict(key,dictionary): #function that returns a value from the dict depending on the key, if not in dict, it updates it
	value = dictionary.get(key,'None')

	if 'None' in value: # if not in dict
		newentry=input(key+' is not in the dictionary, insert a normalised name for it\n')
		DICT[key]=newentry #we update the dictionary
		print('Added '+key+' key as '+newentry)		
		time.sleep(1)
		return newentry	
	else:	
		return (dictionary[key])

def printcontext(cntlines,LINES,character,hlsentence): #function that prints context with a reference sentence, receives the sentence in an array
	limit=Fore.RED + '********************' + Style.RESET_ALL	
	currentstnc=Back.BLUE + '>>>' + Style.RESET_ALL	
	if cntlines>1 and cntlines<len(LINES)-2:
		print(limit)
		print(LINES[cntlines-2]+LINES[cntlines-1])
		print(currentstnc+character+': '+' '.join(hlsentence))					
		print(LINES[cntlines+1]+LINES[cntlines+2])
		print(limit)
		
	elif cntlines<2:
		print(limit)
		print(currentstnc+character+': '+' '.join(hlsentence))					
		print(LINES[cntlines+1]+LINES[cntlines+2])
		print(limit)

	elif cntlines>len(LINES)-2:
		print(limit)
		print(LINES[cntlines-1]+LINES[cntlines-2])
		print(currentstnc+character+': '+' '.join(hlsentence))					
		print(limit)

####
#f=open('characterlistSeason1')
#createdict(f)
####

nlp = spacy.load('en')   ## load the lm

dire='/people/mesa/Desktop/BigBang/transcripts/'
dire2='/people/mesa/Desktop/BigBang/anotations/'

seasonlist=['02','03','04','05','06','07','08','09','10']

seasonumb=sys.argv[1]
##we open the dictionary
with open('chclist', 'r') as content_file:
    content = content_file.read()
DICT = json.loads(content)
##
for fich in listdir(dire+'Season'+seasonumb):
	seasonumb=str(seasonumb)
	band=input('Do you want to process '+fich+'   y/n')
	if 'y' in band:
		f=open(dire+'Season'+seasonumb+'/'+fich,'r') ##we open each file to read it
		LINES=f.readlines()
		f.close()
		f=open(dire+'Season'+seasonumb+'/'+fich,'r') ##we open each file to anotate it
		fanotate=open(dire2+'spacy.anotations/tbbt.season'+seasonumb+'/'+fich,'a')
		cntlines=-1
		for line in f:
			cntlines=cntlines+1
			line_s=line.split(':')
			character=line_s[0]
			if len(line_s)>1:
				sentence=line_s[1]
				sentence=sentence[1:-1]
				if 'Scene' in character: ##we pass scene information					
					pass
				else:
					sentenceproc=nlp(sentence)
					personT=0 #flag to indicate if we found a person in a sentence
					beforename=0
					hlsentence=[] ##array where we will stuck highlited and non highlited words
					personmask=[] ##mask to process the hlsentence, 0 means nothing, 1 means person
					for item in sentenceproc:
						if 'PERSON' in (str(item.ent_type_)):
							wordrplc=Back.YELLOW+Fore.RED + str(item) + Style.RESET_ALL
							hlsentence.append(str(wordrplc))	
							personT=1
							personmask.append('1')						
						else:
							hlsentence.append(str(item)+' ')
							personmask.append('0')	
					###we print the sentence and the context if we found a Person in it or a character not found##
					#Case 1: character not found
					namefound = DICT.get(character,'None')
					if 'None' in namefound:					
						printcontext(cntlines,LINES,character,hlsentence)
						character_norm=lookupdict(character,DICT) #character normalisation
					else: #if there is a person in the sentence we print it
						character_norm=namefound
						if personT:
							printcontext(cntlines,LINES,character,hlsentence)
					###########################################
					#####now we process the person information#
					#we process the character for each line
					cntitem=0
					for item in personmask: #we distinguish between person or not person
						if '1' in item:
							#we check if it is a false positive and we check for composed names
							name=[]
				
							for i in range(cntitem,len(sentenceproc)):
								if '1' in personmask[i]:
									name.append(str(sentenceproc[i]))#we append the names
								else:
									break	
							key=' '.join(name) #we now try to find the name
							nameaux = DICT.get(key,'None')
							if 'None' in nameaux:
								if beforename:
									fanotate.write(str(sentenceproc[cntitem])+' '+str(character_norm)+' person_name '+beforecharacter+' '+ntype+'\n')
								else:	
									flag=input('Is * '+str(sentenceproc[cntitem])+' ... * a name, y/n?')
									if 'n' in flag: #it's a false name
										beforename=0
										fanotate.write(str(sentenceproc[cntitem])+' '+str(character_norm)+' X'+'\n')
										pass
									else:
										if beforename: #if the token before this one was a name we assign the same name
											printentity=entitybefore
										else:									
											printentity=lookupdict(key,DICT)
											ntype=input('Insert type of name, 1/2/3  ')
										entitybefore=printentity
										fanotate.write(str(sentenceproc[cntitem])+' '+str(character_norm)+' person_name '+str(printentity)+' '+ntype+'\n')
										beforename=1
							else: #if we found it in the dictionary
								beforecharacter=str(character_norm)
								if beforename: #we take the same ntype as before
									fanotate.write(str(sentenceproc[cntitem])+' '+str(character_norm)+' person_name '+str(nameaux)+' '+ntype+'\n')
									beforename=1
								else:
									ntype=input('Insert type of name, 1/2/3 for '+str(nameaux)+'  ')
									fanotate.write(str(sentenceproc[cntitem])+' '+str(character_norm)+' person_name '+str(nameaux)+' '+ntype+'\n')
									beforename=1
						else: #if not a person we just write the word and the character who says it
							fanotate.write(str(sentenceproc[cntitem])+' '+str(character_norm)+' X X X'+'\n')
							beforename=0
						cntitem=cntitem+1
					###########################################
					personT=0
					os.system('clear')

		f.close()
		##we update the dictionary
		with open('chclist','w') as fdict:
			json.dump(DICT, fdict)
			fdict.close()



