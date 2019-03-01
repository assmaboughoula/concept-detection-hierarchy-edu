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
import skfuzzy as fuzz



def load_embeddings(embeddings_file):
	# X = np.zeros((N,D))				# concept phrase embeddings matrix
	return

def normalize(M):
	""" normalizes matrix M such that each row sums to 1 """
	return M/M.sum(axis=1)[:,None]


""" Global variables:
"""

MAX_ITER = 1000					# max iterations
CLUSTERS = 20	 				# number of clusters

e = 0.00000001	 				# epsilon error threshold for optimizing the objective function
m = 2							# 1.1 < m < 5 usually good - this approximates cluster quality

similarity_function = "cosine"	# norm function to measure similarity of two vectors
Q = 0.9 						# cluster quality threshold

# N = 7000						# total number of concept phrases
# D = 32 							# concept phrase embedding dimension
# C = np.random.randint(10000, size=(CLUSTERS,D)) 	# cluster centers matrix
# U = normalize(np.random.rand(N, CLUSTERS)) 	# membership matrix: u_ij = probablity of concept i belongs to cluster j
# X = np.random.randint(10000, size=(N,D))


def update_C(U_curr, X, m):
	""" update the cluster centers """

	## U to the power m element wise
	U_m = np.power(U_curr, m)
	
	## transpose U_m , then normalize:
	U_m_norm = normalize(np.transpose(U_m))
	
	## multiply matrix with embeddings X
	C_new = np.matmul(U_m_norm,X)
	
	return C_new

def update_U(C_curr, X, m):
	""" update the membership matrix U """
	X_C = np.power(distance.cdist(X,C_curr,'euclidean'),2.0/(m-1.0))
	U_new = normalize(1.0/X_C)
	return U_new


def calculate_J(U_curr, U_prev):
	""" Calculate the objective function J """
	mse = (np.square(U_curr-U_prev)).mean(axis=None)
	maxerr = np.absolute(U_curr-U_prev).max()
	error = mse
	return error

def c_means(X, C, U, e, m):
	J = 10000
	C_curr = C
	U_curr = U
	Iter=1
	while J>e and Iter< MAX_ITER:
		J_prev = J
		C_curr = update_C(U_curr, X, m)
		U_new = update_U(C_curr, X, m)
		J = calculate_J(U_new, U_curr)
		U_curr = U_new
		Iter+=1
		if Iter==MAX_ITER:
			print ("Reached max iterations without converging! J= ", J)
			break
		if J_prev< J:
			print("J increased by: ", J-J_prev)
			pass
		elif J_prev> J:
			print("J is decreasing: ", J_prev-J)
			pass
	if Iter < MAX_ITER: 
		print("Converged!")
	return C,U 

def layered_fcm(X, CLUSTERS, e, m):
	c = CLUSTERS
	N = np.shape(X)[0]  ## get number of data points
	D = np.shape(X)[1]  ## get dimension of data points
	Centroids = []
	Memberships = []

	while c>1:
		""" Initialize random centroids matrix C """
		C = np.random.randint(np.max(X), size=(c,D))

		""" Initialize random membership matrix U """
		U = normalize(np.random.rand(N, c))

		""" Call c_means for current cluster layer """
		C_final, U_final = c_means(X, C, U, e, m)
		Centroids.append(C_final)
		Memberships.append(U_final)
		
		""" Save results of this layer """
		np.savetxt("Centroids_"+str(c)+".csv", C_final, delimiter=",")
		np.savetxt("Memberships_"+str(c)+".csv", U_final, delimiter=",")

		c -= 1 	## reduce number of clusters for next layer

	return Centroids, Memberships


def cluster_quality(cluster):
	pass

def plot_clusters(X, U):
	
	pass

def test1():
	colors = ['b', 'orange', 'g', 'r', 'c', 'm', 'y', 'k', 'Brown', 'ForestGreen']
	# Define three cluster centers
	centers = [[4, 2],
	           [1, 7],
	           [5, 6]]

	# Define three cluster sigmas in x and y, respectively
	sigmas = [[0.8, 0.3],
	          [0.3, 0.5],
	          [1.1, 0.7]]

	# Generate test data
	np.random.seed(42)  # Set seed for reproducibility
	xpts = np.zeros(1)
	ypts = np.zeros(1)
	labels = np.zeros(1)
	for i, ((xmu, ymu), (xsigma, ysigma)) in enumerate(zip(centers, sigmas)):
	    xpts = np.hstack((xpts, np.random.standard_normal(200) * xsigma + xmu))
	    ypts = np.hstack((ypts, np.random.standard_normal(200) * ysigma + ymu))
	    labels = np.hstack((labels, np.ones(200) * i))

	# Visualize the test data
	fig0, ax0 = plt.subplots()
	for label in range(3):
	    ax0.plot(xpts[labels == label], ypts[labels == label], '.',
	             color=colors[label])
	ax0.set_title('Test data: 200 points x3 clusters.')

	X = np.zeros((601,2))
	X[:,0] = xpts
	X[:,1] = ypts
	print(np.shape(xpts))
	print(np.shape(ypts))
	
	# Centroids, Memberships = layered_fcm(X, 9, 0.005, 2)
	
	fig1, axes1 = plt.subplots(3, 3, figsize=(8, 8))
	alldata = np.vstack((xpts, ypts))
	fpcs = []
	for ncenters, ax in enumerate(axes1.reshape(-1), 2):
	    # cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
	    #     alldata, ncenters, 2, error=0.005, maxiter=1000, init=None)
		fpc = 0.5
		""" Initialize random centroids matrix C """
		C = np.random.randint(np.max(X), size=(ncenters,2))

		""" Initialize random membership matrix U """
		U = normalize(np.random.rand(601, ncenters))

		C_new, U_new = c_means(X, C, U, 0.00005, 2)
		

	    # Store fpc values for later
		fpcs.append(1)

	    # Plot assigned clusters, for each data point in training set
		# print(np.shape(C))
		cluster_membership = np.argmax(np.transpose(U_new), axis=0)
		print(ncenters)
		print(cluster_membership)
		for j in range(ncenters):	
			ax.plot(xpts[cluster_membership == j], ypts[cluster_membership == j], '.', color=colors[j])

		# Mark the center of each fuzzy cluster
		for pt in C:
			ax.plot(pt[0], pt[1], 'rs')

		ax.set_title('Centers = {0}; FPC = {1:.2f}'.format(ncenters, fpc))
		ax.axis('off')

	fig1.tight_layout()	
	plt.show()

	return


def main():
    """ run layered_fcm() """
    N = 70						# total number of concept phrases
    D = 2 							# concept phrase embedding dimension
    X = np.random.randint(10000, size=(N,D))
    # layered_fcm(X, 5, 0.00001, 2)

    test1()
    
    # for arg in sys.argv[1:]:
    #     print arg

if __name__ == "__main__":
    main()