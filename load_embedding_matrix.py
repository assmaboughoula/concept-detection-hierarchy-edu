"""
Functions to load and convert concept embeddings to standard numpy matrix form.
"""

import sys
import numpy as np
import pprint
# import matplotlib.pyplot as plt
# import seaborn as sns; sns.set()
import pandas as pd
# from pandas import Dataframe 

def load_NN_forward_backward(filepath):
	forward = {}
	backward = {}
	bidi = {}
	with open(filepath) as fp:
		line = fp.readline()
		curr_concept = ""
		while line:
			if line.strip() in ['\n','\r\n']:
				line = fp.readline()
				curr_concept = ""
				pass
			elif line.startswith("["):
				raw = line.split()
				current = forward[curr_concept]
				for t in range(len(raw)):
					if t==0:
						current.append(eval(raw[t].replace(',','').replace('[','')))
					elif '[' in raw[t]:
						current = backward[curr_concept]
						current.append(eval(raw[t].replace(',','').replace('[','')))
					else:
						current.append(eval(raw[t].replace(',','').replace(']','')))
				line = fp.readline()
				# print(len(forward[curr_concept]), len(backward[curr_concept]))
				if not 32 == len(forward[curr_concept])== len(backward[curr_concept]):
					print(len(forward[curr_concept]), len(backward[curr_concept]))

			else:
				curr_concept = line.replace('\n','')
				# print(curr_concept)
				if len(curr_concept)>0:
					forward[curr_concept] = []
					backward[curr_concept] = []
				line = fp.readline()
	
	for concept in forward:
		bidi[concept] = forward[concept]
		bidi[concept].extend(backward[concept])
	
	F_embeddings = pd.DataFrame.from_dict(forward)
	F_embeddings.to_csv(path_or_buf=filepath.replace('.txt',"_forward_embeddings.csv"), index=False)
	B_embeddings = pd.DataFrame.from_dict(backward)
	B_embeddings.to_csv(path_or_buf=filepath.replace('.txt',"_backward_embeddings.csv"), index=False)
	BiDi_embeddings = pd.DataFrame.from_dict(bidi)
	BiDi_embeddings.to_csv(path_or_buf=filepath.replace('.txt',"_bidi_embeddings.csv"), index=False)
	print(F_embeddings)
	print(B_embeddings)
	return F_embeddings, B_embeddings, BiDi_embeddings


def load_word2vec(filepath):
	pass


def main():
	load_NN_forward_backward("./concept-embeddings/chunker_bidirectional_embeddings/han_concept_embeddings.txt")
	load_NN_forward_backward("./concept-embeddings/chunker_bidirectional_embeddings/zhai_concept_embeddings.txt")

	load_NN_forward_backward("./concept-embeddings/chunker_bidirectional_embeddings/mooc1_concept_embeddings.txt")
	load_NN_forward_backward("./concept-embeddings/chunker_bidirectional_embeddings/mooc2_concept_embeddings.txt")
	load_NN_forward_backward("./concept-embeddings/chunker_bidirectional_embeddings/mooc3_concept_embeddings.txt")
	load_NN_forward_backward("./concept-embeddings/chunker_bidirectional_embeddings/mooc4_concept_embeddings.txt")


if __name__ == "__main__":
    main()