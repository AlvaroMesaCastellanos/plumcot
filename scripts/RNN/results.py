##script that process results, giving the max accuracy in terms of anotated lines
from os import listdir
import os
import json
import spacy
import numpy as np        
import sys
import linecache
import time
import pandas as pd
import matplotlib.pyplot as plt

dire='/vol/work/mesa/RESULTS/'

for fich in listdir(dire):

	if 'resultsLSTMearlystoppingweightsclusters' in fich:
		data = pd.read_csv(dire+fich, sep=" ", header=None)
		data.columns = ["class", "context", "ncells","pacc" ,"Precision","Rappel","Fbeta=0.1","Fbeta=1","Fbeta=10","Acc","Totallines","%accauto","%accoracle"]
		print(fich)
	
		print('Global ACC:'+str(data['%accauto'].max())+' Local Accuracy:'+str(data['%accoracle'].max())+'\n')
	
	
