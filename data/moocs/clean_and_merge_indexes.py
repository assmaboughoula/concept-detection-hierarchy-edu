import glob, os, re

outputfile = 'quality_phrases_han_zhai.txt'

merge_indexes = open(outputfile, 'w')

zhai = open('zhai_concepts.txt', 'r')

for line in zhai:
	clean = line.split(',')[0].rstrip() #.lower()
	merge_indexes.write(clean+'\n')
zhai.close()

han = open('han_concepts.txt', 'r')
for line in han:
	clean = line.split(',')[0].rstrip() #.lower()
	merge_indexes.write(clean+'\n')
han.close()

merge_indexes.close()
