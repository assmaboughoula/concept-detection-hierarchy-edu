"""
Implementation of c-means fuzzy hierarchicaly clustering (from the bottom up?). Plus Aidan's idea of iterative fuzzy clustering.
"""
import sys
#import pprint
import numpy as np
import scipy
from scipy.spatial import distance
# from sklearn.metrics import pairwise_distances
# from sklearn.metrics.pairwise import pairwise_kernels
# from sklearn.preprocessing import normalize
# import similarity_measures.py
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA as sklearnPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.datasets.samples_generator import make_blobs
from pandas.plotting import parallel_coordinates



def load_embeddings(embeddings_file):
	return

def normalize(M):
	""" normalizes matrix M such that each row sums to 1 """
	return M/M.sum(axis=1)[:,None]


""" Global variables:
"""

D = 32 							# concept phrase embedding dimension
e = 0	 						# epsilon error threshold for optimizing the objective function
m = 2							# 1.1 < m < 5 usually good - this approximates cluster quality
K_MAX = 1000					# max iterations
num_C = 20	 					# number of clusters
N = 7000						# total number of concept phrases
similarity_function = "cosine"	# norm function to measure similarity of two vectors
Q = 0.9 						# cluster quality threshold
C = np.random.randint(10000, size=(num_C,D)) 	# cluster centers matrix
U = normalize(np.random.rand(N, num_C)) 	# membership matrix: u_ij = probablity of concept i belongs to cluster j
# X = np.zeros((N,D))				# concept phrase embeddings matrix
X = np.random.randint(10000, size=(N,D))


def update_C(U_curr, X):
	""" update the cluster centers """

	## U to the power m element wise, then normalize
	U_m = np.power(U_curr, m)
	U_m_norm = normalize(U_m)

	## multiply matrix with embeddings X
	C_curr = np.matmul(np.transpose(U_m_norm),X)
	return C_curr

def update_U(C_curr, X):
	""" update the membership matrix U """
	X_C = np.power(distance.cdist(C_curr,X,'euclidean'),2.0/(m-1.0))
	X_C = normalize(X_C)
	U_new = normalize(np.transpose(1.0/X_C))
	return U_new


def calculate_J(U_curr, U_prev):
	""" Calculate the objective function J """
	return np.absolute(U_curr-U_prev).max()

def c_means(X, C, U):
	J = 1
	C_curr = C
	U_curr = U
	k=1
	while J>e and k< K_MAX:
		J_prev = J
		C_curr = update_C(U_curr, X)
		U_new = update_U(C_curr, X)
		J = calculate_J(U_new, U_curr)
		U_curr = U_new
		k+=1
		if k==K_MAX:
			print ("Reached max iterations without converging! J= ", J)
		if J_prev< J:
			print("J increased by: ", J-J_prev)
		elif J_prev> J:
			print("J is decreasing: ", J)
	print(X,C_curr, U_curr)
	print("Converged!")


def cluster_quality(cluster):
	pass

def plot_clusters(embeddings, clusters_matrix):
	pass

def main():
    # run c_means()
    c_means(X, C, U)
    # for arg in sys.argv[1:]:
    #     print arg

if __name__ == "__main__":
    main()