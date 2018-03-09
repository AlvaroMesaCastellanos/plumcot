from os import listdir
import os
directory='/people/mesa/Desktop/BigBang/otros/'
directory2='/people/mesa/Desktop/BigBang/'
for files in listdir(directory):
	f=open(directory+files,'r')
	f2=open(directory2+files[:-1],'a')
	for line in f:
		if len(line)>1:
			f2.write(line)
		else:
			pass
