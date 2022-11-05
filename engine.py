import os
import numpy as np
import statistics
import matplotlib.pyplot as plt
import cclustering.cclustering_cpu as c_cpu
import cclustering.cclustering_gpu as c_gpu
import cclustering.utils as utils
import data_plot
import utility, histogram_clustering_hierarchical, gaussian_mixture_model
from rich.console import Console
from utility import numberOfBinsSturgesRule, numberOfBinsFreedmanDiaconisRuleModified

os.environ["KMP_WARNINGS"] = "FALSE" 

plt.style.use('ggplot')
console = Console()

# constants
PI = np.pi
PI = np.float32(PI)

'''CIRCLE CLUSTERING'''
def CircleClustering(samples, labels = None, n_dataset = None):

    saver_plots = []

    # CPU 
    numberOfSamplesInTheDataset = samples.shape[0]
    theta = 2 * PI * np.random.rand(numberOfSamplesInTheDataset)
    theta_method1 = np.copy(theta)
    theta_method2 = np.copy(theta)
    theta_method3 = np.copy(theta)
    
    matrixOfWeights, S, C = c_cpu.computing_weights(samples, theta_method1, cosine = False)
    Snew, Cnew = c_cpu.computing_weights_memory_aware(samples, theta_method2, cosine = False)

    print(utils.areEqual(Snew, S, 0.000001))

    theta_method1 = c_cpu.loop(matrixOfWeights, theta_method1, S, C, eps = 0.001)
    theta_method2 = c_cpu.loop_memory_aware(samples, theta_method2, S, C, eps = 0.001)
    
    # GPU -> the gpu needs to enter in action when the number of samples is to big for the RAM
    
    
    # //////////////////////////////////////////////////////////////////
    # PLOTTING PCA
    # data_plot.doPCA(samples, labels, n_dataset)

    # PLOTTING THE THETA
    saver_plots.append(data_plot.plot_circle(theta))

    hist, bins = utility.histogram(theta, nbins=numberOfBinsFreedmanDiaconisRuleModified(theta))

    # PLOTTING THE SCATTER
    saver_plots.append(data_plot.plot_scatter(hist, bins, mode=2))
    # data_plot.plot_hist(hist, bins)
    # data_plot.plot_linespace(theta)
    # //////////////////////////////////////////////////////////////////

    '''
    # smoothing 
    # smooth values with average of ten values
    # we are interested in the hist values because they represent the values to divide
    hist_smoothed_weighted = smoothing_detection.smooth_weighted(hist)
    data_plot.plot_scatter(hist_smoothed_weighted, bins, mode=2)
    data_plot.plot_hist(hist_smoothed_weighted, bins)
    '''

    clusters, thetaLabels, centroids = histogram_clustering_hierarchical.hierarchicalDetectionOfClusters(hist, bins, samples, theta, saver_plots)
    # gaussian_mixture_model.mixtureOfGaussiansManual(len(clusters), bins, theta)
    # gaussianMixtureLabels = gaussian_mixture_model.mixtureOfGaussiansAutomatic(len(clusters), bins, samples, theta)

    # print(clusters)

    # PLOTTING THE THETA WITH COLOURS
    saver_plots.append(data_plot.plot_circle(theta, thetaLabels))

    # return (thetaLabels, gaussianMixtureLabels)
    return thetaLabels + 1, saver_plots