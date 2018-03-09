#This script is intended to parse html content and retrieve the dialogues of a TV-serie

import urllib.request #we use this library to retrieve html content
from bs4 import BeautifulSoup #library for parsing the content
import numpy as np

opener = urllib.request.FancyURLopener({})
root="https://bigbangtrans.wordpress.com/"
url = root
##opening the list of episodes
f=open("/people/mesa/Desktop/BigBang/listofepisodes")
for line in f:	
	episodeurl=(root+line.lower()) #once we have the url of the episode we parse html content
	fp = urllib.request.urlopen(episodeurl)
	content=fp.read()
	soup = BeautifulSoup(content, 'html.parser')
	fepisode=open("/people/mesa/Desktop/BigBang/"+line,'a')	
	for elemento in (soup.find_all('p')):
		string=elemento.get_text()
		if ":" in string:
			fepisode.write(string+"\n")
		else:
			pass
	


