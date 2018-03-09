from os import listdir
import os
import json
import spacy   
import numpy as np        
import sys
import linecache
import time
import math
import keras
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.models import load_model
from sklearn.metrics import confusion_matrix





resultsdire='/people/mesa/Desktop/finalresults/'


fig = plt.figure()
fig.suptitle("Results on accuracy", fontsize='x-large')
fig.subplots_adjust(hspace=0.1, wspace=0.1)

fig2 = plt.figure()
fig2.suptitle("Precision on first class", fontsize='x-large')
fig2.subplots_adjust(hspace=0.1, wspace=0.1)

fig3 = plt.figure()
fig3.suptitle("Recall on first class", fontsize='x-large')
fig3.subplots_adjust(hspace=0.1, wspace=0.1)

fig4 = plt.figure()
fig4.suptitle("FScore first class beta=0.1", fontsize='x-large')
fig4.subplots_adjust(hspace=0.1, wspace=0.1)

fig5 = plt.figure()
fig5.suptitle("FScore first class beta=1", fontsize='x-large')
fig5.subplots_adjust(hspace=0.1, wspace=0.1)

fig6 = plt.figure()
fig6.suptitle("FScore first class beta=10", fontsize='x-large')
fig6.subplots_adjust(hspace=0.1, wspace=0.1)

##SECOND CLASS##

fig7 = plt.figure()
fig7.suptitle("Precision on second class", fontsize='x-large')
fig7.subplots_adjust(hspace=0.1, wspace=0.1)

fig8 = plt.figure()
fig8.suptitle("Recall on second class", fontsize='x-large')
fig8.subplots_adjust(hspace=0.1, wspace=0.1)

fig9 = plt.figure()
fig9.suptitle("FScore second class beta=0.1", fontsize='x-large')
fig9.subplots_adjust(hspace=0.1, wspace=0.1)

fig10 = plt.figure()
fig10.suptitle("FScore second class beta=1", fontsize='x-large')
fig10.subplots_adjust(hspace=0.1, wspace=0.1)

fig11 = plt.figure()
fig11.suptitle("FScore second class beta=10", fontsize='x-large')
fig11.subplots_adjust(hspace=0.1, wspace=0.1)

##SECOND CLASS BEF##
fig12 = plt.figure()
fig12.suptitle("Precision on second class BEF", fontsize='x-large')
fig12.subplots_adjust(hspace=0.1, wspace=0.1)

fig13 = plt.figure()
fig13.suptitle("Recall on second class BEF", fontsize='x-large')
fig13.subplots_adjust(hspace=0.1, wspace=0.1)

fig14 = plt.figure()
fig14.suptitle("FScore second class BEF beta=0.1", fontsize='x-large')
fig14.subplots_adjust(hspace=0.1, wspace=0.1)

fig15 = plt.figure()
fig15.suptitle("FScore second class BEF beta=1", fontsize='x-large')
fig15.subplots_adjust(hspace=0.1, wspace=0.1)

fig16 = plt.figure()
fig16.suptitle("FScore second class BEF beta=10", fontsize='x-large')
fig16.subplots_adjust(hspace=0.1, wspace=0.1)

##SECOND CLASS AFT##
fig17 = plt.figure()
fig17.suptitle("Precision on second class AFT", fontsize='x-large')
fig17.subplots_adjust(hspace=0.1, wspace=0.1)

fig18 = plt.figure()
fig18.suptitle("Recall on second class AFT", fontsize='x-large')
fig18.subplots_adjust(hspace=0.1, wspace=0.1)

fig19 = plt.figure()
fig19.suptitle("FScore second class AFT beta=0.1", fontsize='x-large')
fig19.subplots_adjust(hspace=0.1, wspace=0.1)

fig20 = plt.figure()
fig20.suptitle("FScore second class AFT beta=1", fontsize='x-large')
fig20.subplots_adjust(hspace=0.1, wspace=0.1)

fig21 = plt.figure()
fig21.suptitle("FScore second class AFT beta=10", fontsize='x-large')
fig21.subplots_adjust(hspace=0.1, wspace=0.1)

count1class=1 ##indexes for plotting
count2class=1
count2classplus=1
globalclass=1
###
for fich in listdir(resultsdire):

	experimenttype=fich.split('.')[1]
	
	##DATA LOADING

	data = pd.read_csv(resultsdire+fich, sep=" ", header=None)
	data.columns = ["class", "context", "ncells", "Precision","Rappel","Fbeta=0.1","Fbeta=1","Fbeta=10","Acc","Totallines","%acc"]
	#we put cells in 1....10 instead of pow2	
	data['ncells']=data['ncells'].apply(np.log2)
	data['ncells'].astype('int')
	
	##PLOTTING ACCURACY##
	ax=fig.add_subplot(2,4,int(globalclass))
	
	d1=data[['context','ncells','%acc']][::3][:].values
	
	x=np.zeros((10,10))

	for row in d1:
		r=row.astype('float64')
		x[int(r[1]-1)][int(r[0]-1)]=r[2]
	
	maxacc=x.max()*100
	maxacc=str(maxacc)

	ax.set_title(experimenttype+' '+maxacc[0:5]+'%',fontsize='smaller')

	ax.imshow(x, interpolation='None',vmin=0,vmax=0.2)

	if '4' in fich: ##4classes

		##WE COLLECT ALL THE DATA##
		precision_1class=data[['context','ncells','Precision']][::4][:].values
		precision_2classbef=data[['context','ncells','Precision']][1::4][:].values
		precision_2classaft=data[['context','ncells','Precision']][2::4][:].values

		rappel_1class=data[['context','ncells','Rappel']][::4][:].values
		rappel_2classbef=data[['context','ncells','Rappel']][1::4][:].values
		rappel_2classaft=data[['context','ncells','Rappel']][2::4][:].values
		
		beta01_1class=data[['context','ncells','Fbeta=0.1']][::4][:].values
		beta1_1class=data[['context','ncells','Fbeta=1']][::4][:].values
		beta10_1class=data[['context','ncells','Fbeta=10']][::4][:].values

		beta01_2classbef=data[['context','ncells','Fbeta=0.1']][1::4][:].values
		beta1_2classbef=data[['context','ncells','Fbeta=1']][1::4][:].values
		beta10_2classbef=data[['context','ncells','Fbeta=10']][1::4][:].values

		beta01_2classaft=data[['context','ncells','Fbeta=0.1']][2::4][:].values
		beta1_2classaft=data[['context','ncells','Fbeta=1']][2::4][:].values
		beta10_2classaft=data[['context','ncells','Fbeta=10']][2::4][:].values

		##WE PLOT THE DATA##
		#precision1class
		ax2=fig2.add_subplot(2,4,int(count1class))
		x=np.zeros((10,10))
		for row in precision_1class:
			r=row.astype('float64')
			x[int(r[1]-1)][int(r[0]-1)]=r[2]
	
		maxvalue=x.max()*100
		maxvalue=str(maxvalue)
	
		ax2.set_title(experimenttype+' '+maxvalue[0:5]+'%',fontsize='smaller',weight='bold')
		ax2.imshow(x, interpolation='None',vmin=0,vmax=0.2)
		
		#recall1class
		ax3=fig3.add_subplot(2,4,int(count1class))
		x=np.zeros((10,10))
		for row in rappel_1class:
			r=row.astype('float64')
			x[int(r[1]-1)][int(r[0]-1)]=r[2]
	
		maxvalue=x.max()*100
		maxvalue=str(maxvalue)
	
		ax3.set_title(experimenttype+' '+maxvalue[0:5]+'%',fontsize='smaller',weight='bold')
		ax3.imshow(x, interpolation='None',vmin=0,vmax=0.2)

		#fscore01
		ax4=fig4.add_subplot(2,4,int(count1class))
		x=np.zeros((10,10))
		for row in beta01_1class:
			r=row.astype('float64')
			x[int(r[1]-1)][int(r[0]-1)]=r[2]
	
		maxvalue=x.max()*100
		maxvalue=str(maxvalue)
	
		ax4.set_title(experimenttype+' '+maxvalue[0:5]+'%',fontsize='smaller',weight='bold')
		ax4.imshow(x, interpolation='None',vmin=0,vmax=0.2)

		#fscore1

		ax5=fig5.add_subplot(2,4,int(count1class))
		x=np.zeros((10,10))
		for row in beta1_1class:
			r=row.astype('float64')
			x[int(r[1]-1)][int(r[0]-1)]=r[2]
	
		maxvalue=x.max()*100
		maxvalue=str(maxvalue)
	
		ax5.set_title(experimenttype+' '+maxvalue[0:5]+'%',fontsize='smaller',weight='bold')
		ax5.imshow(x, interpolation='None',vmin=0,vmax=0.2)

		#fscore10

		ax6=fig6.add_subplot(2,4,int(count1class))
		x=np.zeros((10,10))
		for row in beta10_1class:
			r=row.astype('float64')
			x[int(r[1]-1)][int(r[0]-1)]=r[2]
	
		maxvalue=x.max()*100
		maxvalue=str(maxvalue)
	
		ax6.set_title(experimenttype+' '+maxvalue[0:5]+'%',fontsize='smaller',weight='bold')
		ax6.imshow(x, interpolation='None',vmin=0,vmax=0.2)

		
		###SECOND CLASS
		##BEF#######
		#precision2classbef
		ax12=fig12.add_subplot(2,2,int(count2classplus))
		x=np.zeros((10,10))
		for row in precision_2classbef:
			r=row.astype('float64')
			x[int(r[1]-1)][int(r[0]-1)]=r[2]
	
		maxvalue=x.max()*100
		maxvalue=str(maxvalue)
	
		ax12.set_title(experimenttype+' '+maxvalue[0:5]+'%',fontsize='smaller',weight='bold')
		ax12.imshow(x, interpolation='None',vmin=0,vmax=0.2)
		
		#recall1class
		ax13=fig13.add_subplot(2,2,int(count2classplus))
		x=np.zeros((10,10))
		for row in rappel_2classbef:
			r=row.astype('float64')
			x[int(r[1]-1)][int(r[0]-1)]=r[2]
	
		maxvalue=x.max()*100
		maxvalue=str(maxvalue)
	
		ax13.set_title(experimenttype+' '+maxvalue[0:5]+'%',fontsize='smaller',weight='bold')
		ax13.imshow(x, interpolation='None',vmin=0,vmax=0.2)

		#fscore01
		ax14=fig14.add_subplot(2,2,int(count2classplus))
		x=np.zeros((10,10))
		for row in beta01_2classbef:
			r=row.astype('float64')
			x[int(r[1]-1)][int(r[0]-1)]=r[2]
	
		maxvalue=x.max()*100
		maxvalue=str(maxvalue)
	
		ax14.set_title(experimenttype+' '+maxvalue[0:5]+'%',fontsize='smaller',weight='bold')
		ax14.imshow(x, interpolation='None',vmin=0,vmax=0.2)

		#fscore1

		ax15=fig15.add_subplot(2,2,int(count2classplus))
		x=np.zeros((10,10))
		for row in beta1_2classbef:
			r=row.astype('float64')
			x[int(r[1]-1)][int(r[0]-1)]=r[2]
	
		maxvalue=x.max()*100
		maxvalue=str(maxvalue)
	
		ax15.set_title(experimenttype+' '+maxvalue[0:5]+'%',fontsize='smaller',weight='bold')
		ax15.imshow(x, interpolation='None',vmin=0,vmax=0.2)

		#fscore10

		ax16=fig16.add_subplot(2,2,int(count2classplus))
		x=np.zeros((10,10))
		for row in beta10_2classbef:
			r=row.astype('float64')
			x[int(r[1]-1)][int(r[0]-1)]=r[2]
	
		maxvalue=x.max()*100
		maxvalue=str(maxvalue)
	
		ax16.set_title(experimenttype+' '+maxvalue[0:5]+'%',fontsize='smaller',weight='bold')
		ax16.imshow(x, interpolation='None',vmin=0,vmax=0.2)


		##AFT
				#precision2class
		ax17=fig17.add_subplot(2,2,int(count2classplus))
		x=np.zeros((10,10))
		for row in precision_2classaft:
			r=row.astype('float64')
			x[int(r[1]-1)][int(r[0]-1)]=r[2]
	
		maxvalue=x.max()*100
		maxvalue=str(maxvalue)
		
		ax17.set_title(experimenttype+' '+maxvalue[0:5]+'%',fontsize='smaller',weight='bold')
		ax17.imshow(x, interpolation='Gaussian',vmin=0,vmax=0.2)
		
		#recall1class
		ax18=fig18.add_subplot(2,2,int(count2classplus))
		x=np.zeros((10,10))
		for row in rappel_2classaft:
			r=row.astype('float64')
			x[int(r[1]-1)][int(r[0]-1)]=r[2]
	
		maxvalue=x.max()*100
		maxvalue=str(maxvalue)
	
		ax18.set_title(experimenttype+' '+maxvalue[0:5]+'%',fontsize='smaller',weight='bold')
		ax18.imshow(x, interpolation='None',vmin=0,vmax=0.2)

		#fscore01
		ax19=fig19.add_subplot(2,2,int(count2classplus))
		x=np.zeros((10,10))
		for row in beta01_2classaft:
			r=row.astype('float64')
			x[int(r[1]-1)][int(r[0]-1)]=r[2]
	
		maxvalue=x.max()*100
		maxvalue=str(maxvalue)
	
		ax19.set_title(experimenttype+' '+maxvalue[0:5]+'%',fontsize='smaller',weight='bold')
		ax19.imshow(x, interpolation='None',vmin=0,vmax=0.2)

		#fscore1

		ax20=fig20.add_subplot(2,2,int(count2classplus))
		x=np.zeros((10,10))
		for row in beta1_2classaft:
			r=row.astype('float64')
			x[int(r[1]-1)][int(r[0]-1)]=r[2]
	
		maxvalue=x.max()*100
		maxvalue=str(maxvalue)
	
		ax20.set_title(experimenttype+' '+maxvalue[0:5]+'%',fontsize='smaller',weight='bold')
		ax20.imshow(x, interpolation='None',vmin=0,vmax=0.2)

		#fscore10

		ax21=fig21.add_subplot(2,2,int(count2classplus))
		x=np.zeros((10,10))
		for row in beta10_2classaft:
			r=row.astype('float64')
			x[int(r[1]-1)][int(r[0]-1)]=r[2]
	
		maxvalue=x.max()*100
		maxvalue=str(maxvalue)
	
		ax21.set_title(experimenttype+' '+maxvalue[0:5]+'%',fontsize='smaller',weight='bold')
		ax21.imshow(x, interpolation='None',vmin=0,vmax=0.2)

		count1class=count1class+1
		count2classplus=count2classplus+1
	else: #3classes

		##WE COLLECT ALL THE DATA##
		precision_1class=data[['context','ncells','Precision']][::3][:].values
		precision_2class=data[['context','ncells','Precision']][1::3][:].values
		
		rappel_1class=data[['context','ncells','Rappel']][::3][:].values
		rappel_2class=data[['context','ncells','Rappel']][1::3][:].values
		
		beta01_1class=data[['context','ncells','Fbeta=0.1']][::3][:].values
		beta1_1class=data[['context','ncells','Fbeta=1']][::3][:].values
		beta10_1class=data[['context','ncells','Fbeta=10']][::3][:].values
	
		beta01_2class=data[['context','ncells','Fbeta=0.1']][1::3][:].values
		beta1_2class=data[['context','ncells','Fbeta=1']][1::3][:].values
		beta10_2class=data[['context','ncells','Fbeta=10']][1::3][:].values
		
		##WE PLOT THE DATA##
	
		#precision1class
		ax2=fig2.add_subplot(2,4,int(count1class))
		x=np.zeros((10,10))
		for row in precision_1class:
			r=row.astype('float64')
			x[int(r[1]-1)][int(r[0]-1)]=r[2]
	
		maxvalue=x.max()*100
		maxvalue=str(maxvalue)
	
		ax2.set_title(experimenttype+' '+maxvalue[0:5]+'%',fontsize='smaller',weight='bold')
		ax2.imshow(x, interpolation='None',vmin=0,vmax=0.2)

		#recall1class
		ax3=fig3.add_subplot(2,4,int(count1class))
		x=np.zeros((10,10))
		for row in rappel_1class:
			r=row.astype('float64')
			x[int(r[1]-1)][int(r[0]-1)]=r[2]
	
		maxvalue=x.max()*100
		maxvalue=str(maxvalue)
	
		ax3.set_title(experimenttype+' '+maxvalue[0:5]+'%',fontsize='smaller',weight='bold')
		ax3.imshow(x, interpolation='None',vmin=0,vmax=0.2)

		#fscore01
		ax4=fig4.add_subplot(2,4,int(count1class))
		x=np.zeros((10,10))
		for row in beta01_1class:
			r=row.astype('float64')
			x[int(r[1]-1)][int(r[0]-1)]=r[2]
	
		maxvalue=x.max()*100
		maxvalue=str(maxvalue)
	
		ax4.set_title(experimenttype+' '+maxvalue[0:5]+'%',fontsize='smaller',weight='bold')
		ax4.imshow(x, interpolation='None',vmin=0,vmax=0.2)

		#fscore1

		ax5=fig5.add_subplot(2,4,int(count1class))
		x=np.zeros((10,10))
		for row in beta1_1class:
			r=row.astype('float64')
			x[int(r[1]-1)][int(r[0]-1)]=r[2]
	
		maxvalue=x.max()*100
		maxvalue=str(maxvalue)
	
		ax5.set_title(experimenttype+' '+maxvalue[0:5]+'%',fontsize='smaller',weight='bold')
		ax5.imshow(x, interpolation='None',vmin=0,vmax=0.2)

		#fscore10

		ax6=fig6.add_subplot(2,4,int(count1class))
		x=np.zeros((10,10))
		for row in beta10_1class:
			r=row.astype('float64')
			x[int(r[1]-1)][int(r[0]-1)]=r[2]
	
		maxvalue=x.max()*100
		maxvalue=str(maxvalue)
	
		ax6.set_title(experimenttype+' '+maxvalue[0:5]+'%',fontsize='smaller',weight='bold')
		ax6.imshow(x, interpolation='None',vmin=0,vmax=0.2)

		#######2CLASS##############

		#precision2class
		ax7=fig7.add_subplot(2,2,int(count2class))
		x=np.zeros((10,10))
		for row in precision_2class:
			r=row.astype('float64')
			x[int(r[1]-1)][int(r[0]-1)]=r[2]
	
		maxvalue=x.max()*100
		maxvalue=str(maxvalue)
	
		ax7.set_title(experimenttype+' '+maxvalue[0:5]+'%',fontsize='smaller',weight='bold')
		ax7.imshow(x, interpolation='None')

		#recall1class
		ax8=fig8.add_subplot(2,2,int(count2class))
		x=np.zeros((10,10))
		for row in rappel_2class:
			r=row.astype('float64')
			x[int(r[1]-1)][int(r[0]-1)]=r[2]
	
		maxvalue=x.max()*100
		maxvalue=str(maxvalue)
	
		ax8.set_title(experimenttype+' '+maxvalue[0:5]+'%',fontsize='smaller',weight='bold')
		ax8.imshow(x, interpolation='None')

		#fscore01
		ax9=fig9.add_subplot(2,2,int(count2class))
		x=np.zeros((10,10))
		for row in beta01_2class:
			r=row.astype('float64')
			x[int(r[1]-1)][int(r[0]-1)]=r[2]
	
		maxvalue=x.max()*100
		maxvalue=str(maxvalue)
	
		ax9.set_title(experimenttype+' '+maxvalue[0:5]+'%',fontsize='smaller',weight='bold')
		ax9.imshow(x, interpolation='None')

		#fscore1

		ax10=fig10.add_subplot(2,2,int(count2class))
		x=np.zeros((10,10))
		for row in beta1_2class:
			r=row.astype('float64')
			x[int(r[1]-1)][int(r[0]-1)]=r[2]
	
		maxvalue=x.max()*100
		maxvalue=str(maxvalue)
	
		ax10.set_title(experimenttype+' '+maxvalue[0:5]+'%',fontsize='smaller',weight='bold')
		ax10.imshow(x, interpolation='None')

		#fscore10

		ax11=fig11.add_subplot(2,2,int(count2class))
		x=np.zeros((10,10))
		for row in beta10_2class:
			r=row.astype('float64')
			x[int(r[1]-1)][int(r[0]-1)]=r[2]
	
		maxvalue=x.max()*100
		maxvalue=str(maxvalue)
	
		ax11.set_title(experimenttype+' '+maxvalue[0:5]+'%',fontsize='smaller',weight='bold')
		ax11.imshow(x, interpolation='None')


		
		count1class=count1class+1
		count2class=count2class+1

	globalclass=globalclass+1
plt.show()
