"""
Implementation of layered fuzzy c-means clustering using skfuzzy library.
"""
from __future__ import division, print_function
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import skfuzzy as fuzz
import seaborn as sns; sns.set()
from sklearn.datasets.samples_generator import make_blobs
import pandas as pd
import scipy
import operator
from scipy.spatial import distance
from scipy.spatial import ConvexHull
import json
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

""" Global variables:
"""

MAX_ITER = 10000				# max iterations
e = 0.005		 				# epsilon error threshold for optimizing the objective function
m = 2							# 1.1 < m < 5 usually good - this approximates cluster quality


def load_embeddings(embeddings_csv_file):
	"""
	
	"""
	data = pd.read_csv(embeddings_csv_file)
	headers = data.columns.values 
	return data.to_numpy(), headers

def layered_fcm(X, CLUSTERS, e, m):
	"""
	Inputs:
		X: DxN matrix of data point embeddings
		CLUSTERS: Initial layer number of clusters
			default = N/2
		N: number of data points
		D: dimension of data points 
		e: error tolerance
		m: fuzziness
	
	Output:
		Centers : dictionary of center matrices
			Keys = layers
			Values = C[layer]xD matrix
		Membership : dictionary of membership matrices
			Keys = layers
			Values = C[layer]xN
		fpc_vals: list of fpc values for each layer
	"""
	centers = {}
	memberships = {}
	u_inits = []
	distances = []
	obj_fs = []
	iters_run = []
	fpc_vals = {}

	for i in range(CLUSTERS, 0, -1):
		cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(X, i, m, error=e, maxiter=MAX_ITER, init=None)
		centers[i] = cntr
		memberships[i]=u
		u_inits.append(u0)
		distances.append(d)
		obj_fs.append(jm)
		iters_run.append(p)
		fpc_vals[i]=fpc
	
	# np.save('centers_dict.npy',centers)
	# np.save('memberships_dict.npy',memberships)

	return centers, memberships, u_inits, distances, obj_fs, iters_run, fpc_vals

def retain_clusters(X, centers, memberships, p_toler, q_toler):
	"""
	This function does two things:
		1) Defines cluster borders in each layer according to min_p for that layer
			Data points can belong to 0 or >1 clusters in each layer
		2) Decides which layers to retain based on the layer's cluster quality measure (Calinski Harabaz Index)
			Retains all layers with HC_score >= max_HC/2
	"""
	retained_layer_clusters = {}
	layer_quality = {}
	total = 0

	""" Define cluster borders according to min_p """
	for layer in memberships:
		total +=1
		""" Get layer membership and centers matrix """
		u = memberships[layer]
		c = centers[layer]

		""" Define a good membership threshold for the layer min_p """
		big_p = np.max(u)
		small_p = np.min(u)
		# min_p = (big_p+small_p)*p_toler
		min_p = 1.001/(np.shape(u)[0])  ## GOOD NUMERATORS: 1.001

		""" Prune layer membership matrix according to min_p """
		
		u[u>= min_p] = 1
		u[u<min_p] = 0

		""" 
		Prune layers according to cluster quality min_q for the pruned membership matrix
		Decide whether to keep layer or not
		"""
		q = cluster_quality(X, c, u)
		retained_layer_clusters[layer] = u
		layer_quality[layer] = q
	
	""" Define a good quality threshold for the layer min_q """
	max_q = np.max(list(layer_quality.values()))
	min_q = max_q*q_toler

	final_layers = {}
	final_qualities = {}

	R = 0
	for layer in retained_layer_clusters:
		layer_q = layer_quality[layer]
		if layer_q >= min_q:
			final_layers[layer] = retained_layer_clusters[layer]
			final_qualities[layer] = layer_quality[layer]/float(max_q)
			R += 1

	# print("retained_layer_clusters: ", retained_layer_clusters)
	# print("final_qualities: ", final_qualities)
	print("R/total = ", R, "/", total)

	return final_layers, final_qualities


def cluster_quality(X, C_layer, U_layer):
	"""
	Calculate the Calinski Harabaz Index for the 'hardened' layer clusters.
	X: DxN
	C_layer: CxD
	U_layer: CxN
	"""
	N = np.shape(X)[1]
	C = np.shape(C_layer)[0]

	W_distances = distance.cdist(C_layer, np.transpose(X), 'euclidean')
	mean_centroid = np.mean(C_layer, axis=0)
	mean_centroid = np.transpose(mean_centroid[:, np.newaxis])
	B_distances = distance.cdist(mean_centroid, C_layer, 'euclidean')

	# print("W_distances: ", np.shape(W_distances))
	# print("B_distances: ", np.shape(B_distances))

	W = np.sum(W_distances)
	B = np.sum(B_distances)/float(C)

	CH_index = (B/W)*((N-C)/(max(C-1.0, 1.0)))

	return CH_index
	

def EC_score(header, final_layers):
	""" 
	This function calculates the total number of clusters each concept in header 
	belongs to accross all final retained layers, 
	and normalizes it by the max number of membership:
	
	EC_score of 1 --> most elemental concept
	EC_score of 0 --> concept doesn't belong to any cluster(s) --> most composite
	"""
	accumulated_clusters = []
	for layer in final_layers:
		u = final_layers[layer]
		accum = np.sum(u, axis=0)
		accumulated_clusters.append(accum)
	
	total_accum_clusters = np.sum(np.array(accumulated_clusters), axis=0)
	print("min EC degree: ", np.min(total_accum_clusters))
	print("max EC degree: ", np.max(total_accum_clusters))
	
	total_accum_clusters = np.interp(total_accum_clusters, (total_accum_clusters.min(), total_accum_clusters.max()), (0, 1))
	total_accum_clusters = total_accum_clusters[:, np.newaxis]
	
	# norm = np.max(total_accum_clusters)
	# total_accum_clusters = total_accum_clusters/float(norm)

	EC_degree = dict(zip(header, total_accum_clusters))
	EC = pd.DataFrame.from_records(EC_degree)
	return EC

def readable_results(header, final_layers, final_qualities):

	results = {}
	N = len(header)
	for l in final_layers:
		u = final_layers[l]
		C = np.shape(u)[0]
		q = final_qualities[l]
		key = "C = "+str(l)+" Q = "+str(q)
		val = [[header[n] for n in range(N) if u[c][n] == 1 ] for c in range(C) if len([header[n] for n in range(N) if u[c][n] == 1 ])>0]
		if len(val)>0:
			results[key] = val

	# readable = pd.DataFrame.from_records(results)
	with open('readable.json', 'w') as fp:
		json.dump(results, fp, indent=4) # sort_keys=True
	return results




def plot_clusters(X, centers, final_layers, final_qualities, header, EC_degree):
	
	""" Perform PCA on data points X to project onto 2d """
	N= np.shape(X)[1]
	L = len(final_layers)
	boxW = np.ceil(np.sqrt(L))
	## Standardize the data:
	X = np.transpose(X)
	X = StandardScaler().fit_transform(X)

	## Dimensionality reduction using PCA:
	pca = PCA(n_components=2)
	principalComponents = pca.fit_transform(X)
	principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])



	## Visualize 2d PCA projection:
	cmap = matplotlib.cm.get_cmap('viridis')
	
	# colors = np.random.rand(N)
	colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
	
	fig1 = plt.figure(figsize = (20,20))  ##figsize = (10,10)
	
	temp = list(final_layers.values())
	temp = [np.sum(e) for e in temp]
	temp = [e for e in temp if e>0] ## get the layers with non-empty clusters
	temp = len(temp)
	boxW = np.ceil(np.sqrt(temp+1))

	ax = fig1.add_subplot(boxW,boxW,1) 
	ax.set_xlabel('PC1')
	ax.set_ylabel('PC2')
	ax.set_title('Concept data pts')
	x = principalDf.loc[:, 'principal component 1']
	y = principalDf.loc[:, 'principal component 2']
	ax.scatter(x, y, c=colors[3], alpha=0.5)  # ax.scatter(x, y, s=area, c=colors, alpha=0.5)
	ax.grid()

	plotted_layer = {}
	i = 2
	for l in final_layers:
		u = final_layers[l]
		c = np.shape(u)[0]

		if np.sum(u)>=1:
			plotted_layer[l] = 1
			layer_plt = fig1.add_subplot(boxW,boxW,i)
			layer_plt.set_title('Layer #'+str(l)+' clusters')

			# centersPCA = pca.fit_transform(centers[l])
			# centers2d = pd.DataFrame(data=centersPCA, columns= ['pc1', 'pc2'])
			# # print("centers2d: ", np.shape(centers2d), centers2d)

			for j in range(c): ## for each cluster in the layer
				x = [elem for elem in np.multiply(u[j], principalDf.loc[:, 'principal component 1']) if elem != 0]
				y = [elem for elem in np.multiply(u[j], principalDf.loc[:, 'principal component 2']) if elem != 0]

				if len(x)<1:
					pass
				else:
					layer_plt.scatter(x, y, c=colors[j%len(colors)], alpha=0.7)  # [cmap(1.0-j/float(c))]*len(x) | [j]*len(x)
				
					p = np.c_[x,y]
					mean = np.mean(p, axis=0)
					d = p-mean
					r = np.max(np.sqrt(d[:,0]**2+d[:,1]**2 ))
					circ = plt.Circle(mean, radius=1.05*r, color=colors[j%len(colors)], alpha=0.2 )
					layer_plt.add_patch(circ)
			i += 1
		else:
			plotted_layer[l] = 0
			pass
	
	plt.savefig("X_scatter.png")
	plt.show()


	""" Plot layer qualities: Layer vs. HC Index with quality threshold line plot """
	
	layers = list(final_qualities.keys())
	del layers[0]
	# print('layers', layers)
	qualities = list(final_qualities.values())
	del qualities[0]
	plotted_bool = list(plotted_layer.values())
	del plotted_bool[0]

	layer_colors = []
	for v in plotted_bool:
		if v==0:
			layer_colors.append('k')
		else:
			layer_colors.append('c')

	fig2 = plt.figure()
	b = fig2.add_subplot(1,1,1) 
	b.set_xlabel('Layers')
	b.set_ylabel('Calinski Harabaz Index')
	b.set_title('Layer Qualities')
	b.scatter(layers, qualities, c=layer_colors, alpha= 0.5)
	b.grid()
	plt.savefig("layer_qual.png")
	plt.show()


	""" Plot EC_degree """
	# header, EC_degree
	EC_degree = EC_degree.sort_values(by=0, ascending=False, axis=1)
	print(EC_degree)
	EC_degree.T.to_csv(path_or_buf="sorted_EC_degrees.csv")
	
	concept_phrases = list(EC_degree)
	# print(concept_phrases)
	x_coords = [x for x in range(1, len(concept_phrases)+1)]
	concept_scores = EC_degree.iloc[0,:].to_numpy()
	
	fig3 = plt.figure()
	c = fig3.add_subplot(1,1,1)
	c.set_xlabel('Concepts')
	c.set_ylabel('EC_score')
	c.set_title('Elemental/Composite concept scores')
	c.scatter(x_coords, concept_scores, c='g', alpha=0.5)

	num_concepts = len(concept_phrases)

	for i, txt in enumerate(concept_phrases):
		if i in [20, num_concepts/2, num_concepts-20]:
			c.annotate(txt, (x_coords[i], concept_scores[i]))
		else:
			pass
	
	plt.savefig("EC_scores.png")
	plt.show()


	

def main():
    
    """ Define data point matrix X """
    # N = 70							# total number of concept phrases
    # D = 2 							# concept phrase embedding dimension
    # X = np.random.randint(1000, size=(D,N))
    # X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)
    # X = np.transpose(X)
    # print(np.shape(X))

    # X = load_embeddings("./concept-embeddings/chunker_bidirectional_embeddings/han_concept_embeddings_forward_embeddings.csv")
    X, header = load_embeddings("./concept-embeddings/chunker_bidirectional_embeddings/han_concept_embeddings_backward_embeddings.csv")
    #print(X)
    print("X shape: ", np.shape(X))

    """ Default parameter values """
    
    e = 0.005		 				# epsilon error threshold for optimizing the objective function
    m = 2							# 1.1 < m < 5 usually good - this approximates cluster quality

    N = np.shape(X)[1]
    CLUSTERS = int(N/10)
    # CLUSTERS = int(np.shape(X)[1]/2)
    p_toler = 0.7 	## GOOD ONES: 0.5
    q_toler = 0.01 	## GOOD ONES: 0.01

    """ Run layered_fcm() """
    centers, memberships, u_inits, distances, obj_fs, iters_run, fpc_vals = layered_fcm(X, CLUSTERS, e, m)
    
    """ Get retained layers and qualities """
    final_layers, final_qualities = retain_clusters(X, centers, memberships, p_toler, q_toler)

    """ Get EC_score for concepts """
    print("header: ", header)
    EC = EC_score(header, final_layers)
    print(EC)
    # EC.to_csv(path_or_buf="han_concepts_EC_degrees.csv")

    """ Get readable results of retained cluster layers and qualities with concept phrases """
    readable = readable_results(header, final_layers, final_qualities)
    # print(readable)
    
    """ Plot each layer and the EC plot """
    plot_clusters(X, centers, final_layers, final_qualities, header, EC)

    # for arg in sys.argv[1:]:
    #     print arg

if __name__ == "__main__":
    main()