#!/usr/bin/python
# Modified from the work by Patrick Crain (pcrain2)

import csv, sys, os, re
from os import listdir
from os.path import isfile, join
import spacy
import argparse

nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner', 'textcat'])
nlp.max_length = 1500000

def preprocess(words):
  #Remove Line breaks
  words = re.sub(r'''-\n''',r"",words)
  #Remove weird fi character
  words = re.sub(r'ﬁ',r'fi', words)
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
  parser = argparse.ArgumentParser(description='Takes in a texfile and outputs the IOB-tagged file')
  parser.add_argument('--input_filename', '-i')
  parser.add_argument('--concept_filename', '-c')
  parser.add_argument('--iob_filename', '-o') # Output file
  args = parser.parse_args()

  INPUTFILE = args.input_filename
  CONCEPTFILE  = args.concept_filename
  IOBFILE = args.iob_filename

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
