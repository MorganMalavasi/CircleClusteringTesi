from importlib.resources import path
import os
import sys
import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity

def numberOfBinsSturgesRule(points):
    n = points.shape[0]
    """
    Sturges rule takes into account the size of the data to decide on the number of bins. 
    The formula for calculating the number of bins is shown below.
    bins = 1 + ceil(log2(n))
    In the above equation ’n’ is the sample size. 
    The larger the size of the sample, the larger would be the number of bins. 
    Ceiling the result of the logarithm ensures the result to be an integer.
    """
    bin_count = int(np.ceil(np.log2(n)) + 1)
    return bin_count

def numberOfBinsFreedmanDiaconisRuleModified(points):
    """
    Freedman-Diaconis rule not only considers the sample size but also considers the spread of the sample.
    bin width= 2 * (IQR(x) / n‾√3)
    bins = ceil(  (max(x) - min(x)) / bin width  )

    In the above equation ‘q3’ stands for third quartile, 
    ‘q1' stands for first quartile and ’n’ stands for sample size. 
    As the sample size increases, the bin width decreases which in turn increases the number of bins.
    """
    norm_dist = pd.Series(points)
    q1 = norm_dist.quantile(0.25)
    q3 = norm_dist.quantile(0.75)
    iqr = q3 - q1
    bin_width = (2 * iqr) / (len(norm_dist) ** (1 / 3))
    bin_count = int(np.ceil((norm_dist.max() - norm_dist.min()) / bin_width))
    
    nbins = pow(bin_count, 2)
    if nbins < 128:
        nbins = 128 

    # print("Number of bins = {0}".format(nbins))
    return nbins


def histogram(theta, nbins=None, verb=True):
    if nbins is None:
        nbins = len(theta)
    # Return evenly spaced numbers over a specified interval.
    # start = 0
    # stop = 2*PI
    # nbins = Number of samples to generate
    bins = np.linspace(0,2*np.pi,nbins)
    h, b = np.histogram(theta, bins)
    return h, b

def averageOfList(lst):
    return sum(lst) / len(lst)

def discretization(array, bins):
    return np.digitize(array, bins)

# remove low percentage
def removeLowPercentage(hist):
    newHist = np.empty(hist.shape[0])
    
    sumOfDifferences = 0
    count = 0
    for i in range(hist.shape[0]-1):
        sumOfDifferences += abs(hist[i+1] - hist[i])
        count += 1
    
    threshold = (sumOfDifferences / count)*2

    for j in range(hist.shape[0]):
        if hist[j] >= threshold:
            newHist[j] = hist[j]
        else:
            newHist[j] = 0
    return newHist

# circular space problem
def rotateHistogram(hist, bins):
    # find min
    minHist = sys.maxsize
    index = 0
    for i in range(hist.shape[0]):
        if hist[i] < minHist:
            minHist = hist[i]
            index = i

    return rotate(hist, index, bins)

def rotate(hist, pivot, bins):
    n = hist.shape[0]
    movement = n - pivot
    newHist = np.roll(hist, movement)
    newBins = np.roll(bins, movement)
    return newHist, newBins, movement


# TODO -> find an heuristic for selecting the value of the percentage to remove
def removeLowPercentageOfNoise(hist):
    newHist = np.empty(hist.shape[0])
    
    maxHeight = max(hist)
    maxHeight_5_percent = maxHeight / 20
    for i in range(hist.shape[0]):
        if hist[i] < maxHeight_5_percent:
            newHist[i] = 0
        else:
            newHist[i] = hist[i]
    
    return newHist
    

# old version
def removeCircularSpace(hist):
    while True:
        pivot = findAnEmptySpace(hist)
        if pivot != -1:
            return rotate(hist, pivot)
        else:
            hist = removeSpace(hist)

def findAnEmptySpace(hist):
    for i in range(hist.shape[0]):
        if hist[i] == 0:
            return i
    return -1

def removeSpace(hist):
    minHist = min(hist)
    newHist = np.empty(hist.shape[0])
        
    for i in range(hist.shape[0]):
        newHist[i] = hist[i] - minHist
    
    return newHist

# smoothing
def smooth_weighted(values):
    """
    Compute the smoothing of a line of values
    Given a line of "values", it is applied an averaging filter for smoothing it

    Parameters
    ----------
    values : ndarray
        1D NumPy array of float dtype representing n-dimensional points on a chart
    smoothing_index : int
        value representing the size of the window for smoothing
        default = 10

    Returns
    -------
    output : ndarray 
        line of "values" smoothed
    """
    output = np.empty([values.shape[0]])

    # define the weight for the gaussian
    smoothing_index = 7
    gaussianWeights = np.array([0.25, 1, 2, 4, 2, 1, 0.25])

    for i in range(values.shape[0]):
        sum = 0.0
        count = 0
        for j in range(smoothing_index):
            x = j - int(smoothing_index/2)
            if (i+x)>=0 and (i+x)<values.shape[0]:
                sum = sum + values[i+x] * gaussianWeights[j]
                count += 1

        output[i] = sum / count

    return output

def kde(bins, theta):
    """
    A histogram is a simple visualization of data where bins are defined, 
    and the number of data points within each bin is tallied

    Here we have used kernel='gaussian'. 
    Mathematically, a kernel is a positive function which is controlled by the bandwidth parameter. 
    Given this kernel form, the density estimate at a point within a group of points is given by:

    pk(y) = sum[1,...n]( K(y - xi; h) )

    The bandwidth here acts as a smoothing parameter, 
    controlling the tradeoff between bias and variance in the result. 
    A large bandwidth leads to a very smooth (i.e. high-bias) density distribution. 
    A small bandwidth leads to an unsmooth (i.e. high-variance) density distribution.
    """
    # - smoothing nr 1 -> KDE 
    model = KernelDensity(bandwidth=(bins[1]/2), kernel='gaussian')
    sample = theta.reshape((len(theta), 1))
    model.fit(sample)

    # values = np.asarray([value for value in range(1, bins.shape[0])])
    values = np.copy(bins)
    values = values.reshape(len(values), 1)
    probabilities = model.score_samples(values)
    probabilities = np.exp(probabilities)

    # data_plot.plot_scatter(probabilities, bins, mode=2)
    
    # plt.hist(sample, bins=bins, density=True)
    # plt.plot(values[:], probabilities)
    # plt.show()

    return probabilities

def windowSize(nbins, theta):
    val128 = 128
    size = 0
    # TODO -> il valore della grandezza della finestra non deve essere dato dal numero di bins 
    # ma da quanto sono rumorosi i dati in ingresso
    while (val128 <= nbins):
        val128 *= 2
        size += 1

    return size



# get the value with more repetition in an array 
def find_max_repeating_number_in_array_using_count(arr):
    
    clusters = -1
    max_rep = 0
    for i in range(len(arr)):
        elementToCheck = arr[i]
        
        counter = 0
        for j in range(len(arr)):
            if elementToCheck == arr[j]:
                counter += 1
        
        if counter > max_rep:
            max_rep = counter
            clusters = elementToCheck
    
    return clusters

def my_get_dataset_names(batteryName):
    nameInitDir = "../clustering-data-v1-1.1.0"
    nameDir = nameInitDir + '/' + batteryName
    listOfFiles = os.listdir(nameDir)

    listOfDatasets = []
    for file in listOfFiles:
        if 'data.gz' in file:
            listOfDatasets.append(file.replace('.data.gz', ''))

    return listOfDatasets


def print_results_single_dataset(info):
    string_best = ""
    string_worst = ""
    for i in info:
        name = i[0]
        if name == "CircleClustering":
            score_rand_circle_clustering = i[1]
            score_mutual_circle_clustering = i[2]

            # find where the algorithm has performed best respect other clustering methodss
            for eachOtherDataset in info:
                if eachOtherDataset[0] != None and eachOtherDataset[0] != "CircleClustering":
                    if score_rand_circle_clustering > eachOtherDataset[1]:
                        string_best = string_best + " " + eachOtherDataset[0] + ","
                    else:
                        string_worst = string_worst + " " + eachOtherDataset[0] + ","

    return "<h4 style=\"padding-left: 45;\">CircleClustering >{0}</h1>\n".format(string_best), "<h4 style=\"padding-left: 45;\">CircleClustering <{0}</h1>\n".format(string_worst)

def print_results_total_datasets(results):
    table_rand_score = build_table_results(results)
    strings = []    
    for algorithm_result in range(1, table_rand_score.shape[1]):
        nameAlgorithmEvaluated = table_rand_score[0, algorithm_result, 0]
        counter = 0
        for i in range(table_rand_score.shape[0]):
            tuple_circle_clustering = table_rand_score[i, 0]
            nameCircleClustering = tuple_circle_clustering[0]
            scoreCircleClustering = tuple_circle_clustering[1]

            tuple_result_alg = table_rand_score[i, algorithm_result]
            name = tuple_result_alg[0]
            score = tuple_result_alg[1]
            if scoreCircleClustering > score:
                counter += 1

        percentual = round((counter/table_rand_score.shape[0]) * 100, 2)
        strings.append("<h3 style=\"padding-left: 45;\">The algorithm beated {0} {1}".format(nameAlgorithmEvaluated, percentual) + "%" + " of the times</h3>\n")
    return strings
        
def build_table_results(results):
    table = []
    for dataset in results:
        nameDSET = dataset[0]
        run = dataset[1]   
        
        results_run = []
        for resultOfARun in run:
            name_algorithm = resultOfARun[0]
            score_rand = resultOfARun[1]
            score_mutual = resultOfARun[2]

            if name_algorithm == None:
                continue
            else:
                results_run.append((name_algorithm, score_rand))
        table.append(results_run)

    return np.array(table) 
        