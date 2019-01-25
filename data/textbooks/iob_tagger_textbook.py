#!/usr/bin/python
# Modified from the work by Patrick Crain (pcrain2)

import csv, sys, os, re
from os import listdir
from os.path import isfile, join
import spacy

INPUTFILE = "./zhai_main.txt"
CONCEPTFILE     = "./zhai_concepts.txt"
IOBFILE     = "./zhai_tb_iob_tags.txt" # Output file

nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner', 'textcat'])

def preprocess(words):
  #Remove Line breaks
  words = re.sub(r'''-\n''',r"",words)
  #Normalize aprostrophes
  words = re.sub(r'''’''',r"'",words)
  #Remove non alphabetic characters
  words = re.sub(r'''[^a-zA-Z\u0391-\u03A9\u03B1-\u03C9\ \-\.]''',r''' ''',words)
  #Collapse sequences of whitespace
  words = re.sub(r'''\s+''',r''' ''',words)
  #Normalize to lowercase
  return [token.lemma_ for token in nlp(words)]

def buildConceptChain(conceptlist,cdict=None):
  if cdict is None:
    cdict = {}
  for concept in conceptlist:
    d = cdict
    for i in range(len(concept)):
      if concept[i] not in d:
        d[concept[i]] = {}
      d = d[concept[i]]
    d["CONCEPT"] = True

  return cdict

def loadAndPreprocessWords(path):
  with open(path,"r") as infile:
    words = infile.read()
  newwords = preprocess(words)
  return newwords

def main():
  conceptlist = []
  with open(CONCEPTFILE,"r") as conceptfile:
    for line in conceptfile:
      conceptlist.append(preprocess(line))

  concepts = buildConceptChain(conceptlist)
  # print(concepts)

  words    = loadAndPreprocessWords(INPUTFILE)
  tags     = ["O" for _ in range(len(words))]
  for i in range(len(words)):
    c = concepts
    n = i
    found = None
    while words[n] in c:
      c = c[words[n]]
      n += 1
      if "CONCEPT" in c:
        found = n
    if found is not None:
      for k in range(i,found):
        tags[k] = "B" if k==i else "I"
      i = found

  with open(IOBFILE,"w") as outfile:
    for i in range(len(tags)):
      line = "{1}\t{0}\n".format(tags[i],words[i])
      #print(line,end="")
      outfile.write(line)

  # print(concepts)

if __name__ == "__main__":
  main()
