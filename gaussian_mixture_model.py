import sys
from random import sample
from tkinter import N
import numpy as np
import matplotlib.pyplot as plt
from itertools import chain
from sklearn.mixture import GaussianMixture
from data_plot import drawMixtureOfGaussians
import metrics
from utility import find_max_repeating_number_in_array_using_count


######################### IMPLEMENTATION OF MIXTURE OF GAUSSIANS MANUALLY ###########################
def mixtureOfGaussiansManual(components, bins, theta):
    
    colors = ["b", "g", "r", "c", "m", "y", "k", "w"]

    k = components
    weights = np.ones((k)) / k 
    means = np.random.choice(theta, k)
    variances = np.random.random_sample(size = k)
    # print(means, variances)

    eps = 1e-8
    steps = 100
    for step in range(100):

        likelihood = []
        for j in range(k):
            likelihood.append(pdf(theta, means[j], np.sqrt(variances[j])))
        likelihood = np.array(likelihood)

        b = []
        # maximization step
        for j in range(k):
            # use the current values for the parameters to evaluate the posterior
            # probabilities of the data to have been generanted by each gaussian    
            b.append((likelihood[j] * weights[j]) / (np.sum([likelihood[i] * weights[i] for i in range(k)], axis=0)+eps))

            # updage mean and variance
            means[j] = np.sum(b[j] * theta) / (np.sum(b[j]+eps))
            variances[j] = np.sum(b[j] * np.square(theta - means[j])) / (np.sum(b[j]+eps))

            # update the weights
            weights[j] = np.mean(b[j])

    plt.figure(figsize=(10, 6))
    axes = plt.gca()
    plt.xlabel("$samples$")
    plt.ylabel("pdf")
    plt.title("Gaussian mixture model")
    plt.scatter(theta, [-0.05] * len(theta), color='navy', s=30, marker=2, label="Train data")

    for i in range(components):
        plt.plot(bins, pdf(bins, means[i], variances[i]), color=colors[i], label="Cluster {0}".format(i+1))
    
    plt.legend(loc='upper left')
    
    # plt.savefig("img_{0:02d}".format(step), bbox_inches='tight')
    plt.show()
    return 

def pdf(data, mean: float, variance: float):
  # A normal continuous random variable.
  s1 = 1/(np.sqrt(2*np.pi*variance))
  s2 = np.exp(-(np.square(data - mean)/(2*variance)))
  return s1 * s2

######################### IMPLEMENTATION OF MIXTURE OF GAUSSIANS WITH SKLEARN ########################

def mixtureOfGaussiansAutomatic(k, bins, samples, theta):
    thetaReshaped = theta.reshape(-1,1)

    # evaluate the best model for the gaussian mixture using bic 
    # we cx\heck in the neighbours of the k found
    n_components_range_lower = range(k-2, k)
    n_components_range_higher = range(k, k+3)
    
    # method1 
    '''
    n_components_range = chain(n_components_range_lower, n_components_range_higher)
    labelsDecisionBasedOnBic = decisionBasedOnBic(n_components_range, thetaReshaped)
    '''

    # method2
    n_components_range = chain(n_components_range_lower, n_components_range_higher)
    labelsdecisionBasedOnMultipleFactors = decisionBasedOnMultipleFactors(n_components_range, samples, thetaReshaped)

    # draw with plot the mixture of Gaussians 
    # drawMixtureOfGaussians(theta, bins, best_gmm)

    return labelsdecisionBasedOnMultipleFactors

def decisionBasedOnBic(n_components_range, thetaReshaped):
    print("----------------")
    lowest_bic = np.infty
    bic = []
    for i in n_components_range:
        if i <= 1:
            continue
        gmm = GaussianMixture(n_components = i)
        gmm.fit(thetaReshaped)
        bic.append(gmm.bic(thetaReshaped))
        print("Nr. components : {0}, bic = {1}".format(i, bic[-1]))
        if bic[-1] < lowest_bic:
            lowest_bic = bic[-1]
            best_gmm = gmm
    print("----------------")
    labels = best_gmm.predict(thetaReshaped)
    return labels

def decisionBasedOnMultipleFactors(n_components_range, samples, thetaReshaped):
    # the decision is taken based on multiple internal validity index
    index_major = []
    index_minor = []
    howManyClusters = []
    dictClustersLabels = {}
    for i in n_components_range:
        if i <= 1:
            continue
        
        gmm = GaussianMixture(n_components=i)
        gmm.fit(thetaReshaped)
        labels = gmm.predict(thetaReshaped)
        dictClustersLabels[i] = labels
        
        score_silhouette = metrics.silhouette(samples, labels)
        score_calinski = metrics.calinski(samples, labels)
        score_dunn = metrics.dunn_fast(samples, labels)
        score_pearson = metrics.pearson(samples, labels)
        index_major.append((i, [score_silhouette, score_calinski, score_dunn, score_pearson]))

        score_bic = gmm.bic(thetaReshaped)
        score_widest_within_cluster_gap = metrics.widest_within_cluster_gap_formula(samples, labels)
        index_minor.append((i, [score_bic, score_widest_within_cluster_gap]))

    computeIndex(index_major, howManyClusters, major_minor=True)
    computeIndex(index_minor, howManyClusters, major_minor=False)
    
    # print(howManyClusters)
    max_repeating_number_in_array = find_max_repeating_number_in_array_using_count(howManyClusters)
    average_of_clusters_index = round(np.average(np.array(howManyClusters)))

    nr_clusters = max_repeating_number_in_array

    return dictClustersLabels.get(nr_clusters)
        
            
def computeIndex(index, howManyClusters, major_minor = True):
    for j in range(len(index[0][1])):
        stack = []
        for i in range(len(index)):
            row = index[i]
            cl = row[0]
            val = row[1][j]
            stack.append((cl, val))
        
        if (major_minor):
            cl = -sys.maxsize
            max_val = -sys.maxsize
            for eachTuple in stack:
                if eachTuple[1] > max_val:
                    max_val = eachTuple[1]
                    cl = eachTuple[0]

            howManyClusters.append(cl)
        
        else:
            cl = sys.maxsize
            min_val = sys.maxsize
            for eachTuple in stack:
                if eachTuple[1] < min_val:
                    min_val = eachTuple[1]
                    cl = eachTuple[0]

            howManyClusters.append(cl)
        
