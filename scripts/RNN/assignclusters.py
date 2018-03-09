#script that assigns clusters to every character who is speaking (assuming a perfect speaker diarization)

from os import listdir
import os
import json
import spacy   
import numpy as np        
import sys
import linecache
import time

dire='/people/mesa/Desktop/BigBang/anotations/spacy.anotations/tbbt.season1/' ##season we will use to assign the clusters


##we create the dictionary in which we will store the clusters, {normalised_person_name:#cluster,....}
clustersdict={}

for ele in listdir(dire):
	f=open(dire+ele,'r')
	for line in f:
		line_s=line.split(' ')
		character=line_s[1]
		##we retrieve the cluster assigned to that speaker
		namefound = clustersdict.get(character,'None')
		if 'None' in str(namefound): #not found, we create a new record
			clustersdict[character]=(len(clustersdict)+1)
		else:
			pass
	

	f.close()
##we save the clustersdict
with open('/people/mesa/Desktop/scripts/clustersS1','w') as fdict:
			json.dump(clustersdict, fdict)
			fdict.close()



