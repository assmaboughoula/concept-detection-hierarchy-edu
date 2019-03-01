"""
Implementation of layered fuzzy c-means clustering using skfuzzy library.
"""
from __future__ import division, print_function
import sys
import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz
import seaborn as sns; sns.set()
from sklearn.datasets.samples_generator import make_blobs

""" Global variables:
"""

MAX_ITER = 1000					# max iterations
CLUSTERS = 20	 				# number of clusters

e = 0.005		 				# epsilon error threshold for optimizing the objective function
m = 2							# 1.1 < m < 5 usually good - this approximates cluster quality


def load_embeddings(embeddings_file):
	# X = np.zeros((N,D))				# concept phrase embeddings matrix
	return

def layered_fcm(X, CLUSTERS, e, m):
	centers = []
	memberships = []
	u_inits = []
	distances = []
	obj_fs = []
	iters_run = []
	fpc_vals = []

	for i in range(CLUSTERS, 0, -1):
		cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(X, i, m, error=e, maxiter=MAX_ITER, init=None)
		centers.append(cntr)
		memberships.append(u)
		u_inits.append(u0)
		distances.append(d)
		obj_fs.append(jm)
		iters_run.append(p)
		fpc_vals.append(fpc)

	return centers, memberships, u_inits, distances, obj_fs, iters_run, fpc_vals

def plot_clusters(X, C, U):
	cluster_membership = np.argmax(U, axis=0)  # Hardening for visualization
	print(cluster_membership)
	plt.scatter(X[0,:], X[1,:], c= cluster_membership, s=50);
	plt.scatter(C[:, 0], C[:, 1], c='black', s=200, alpha=0.5);
	plt.show()

def main():
    
    """ Default parameter values """
    CLUSTERS = 9
    e = 0.005		 				# epsilon error threshold for optimizing the objective function
    m = 2							# 1.1 < m < 5 usually good - this approximates cluster quality

    """ Define data point matrix X """
    # N = 70							# total number of concept phrases
    # D = 2 							# concept phrase embedding dimension
    # X = np.random.randint(1000, size=(D,N))
    X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)
    X = np.transpose(X)
    print(np.shape(X))

    """ Run layered_fcm() """
    centers, memberships, u_inits, distances, obj_fs, iters_run, fpc_vals = layered_fcm(X, CLUSTERS, e, m)
    
    """ Plot each layer and the fpc plot """
    for i in range(CLUSTERS):
    	plot_clusters(X, centers[i],memberships[i])
    plt.plot(np.arange(CLUSTERS,0,-1), np.array(fpc_vals), 'ro')
    plt.show()

    # for arg in sys.argv[1:]:
    #     print arg

if __name__ == "__main__":
    main()