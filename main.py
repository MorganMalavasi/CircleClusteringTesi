import os
from multiprocessing import Process, Queue
import numpy as np
import matplotlib.pyplot as plt
from rich.console import Console
import engine
import clustbench
from utility import my_get_dataset_names
import sklearn.metrics as metrics
import sklearn.cluster as clustering_sklearn
import genieclust
import data_plot

os.environ["KMP_WARNINGS"] = "FALSE" 

plt.style.use('ggplot')
console = Console()

def main():
    
    # dataset = data_generation.createDatasets(6)
    # samples, labels, n_dataset = dataset[0], dataset[1], dataset[2]

    """
        Here we build a mega procedure for testing our clustering algorithms.
        We will benchmark and compare the following algorithms:
            - CircleClustering
            - ...

        For the testing we will use the framework 'Clustering Benchmarks' by Marek Gagolewski
        with the batteries from the big dataset always from Marek G.

        In paricular the DB is composed of 9 main groups of dataset (batteries).
        ['fcps', 'g2mg', 'graves', 'h2mg', 'mnist', 'other', 'sipu', 'uci', 'wut']
        Most important, each group is composed of a big number of different datasets
        
        For comparing the differences and enstablish the best algorithms among all
        we will use some external validity indeces. 
        [...]

    """

    data_path = os.path.join("../clustering-data-v1-1.1.0")
    

    batteries_names = clustbench.get_battery_names(path=data_path)
    print(batteries_names)
    # loop on all the groups of datasets
    for eachBatteryName in batteries_names:
        print(eachBatteryName)
        # loop on each dataset in the battery
        battery = my_get_dataset_names(eachBatteryName)
        print(battery)
        all_figures = []
        for eachDatasetName in battery:
            benchmark = clustbench.load_dataset(eachBatteryName, eachDatasetName, path=data_path)
            X = benchmark.data
            y_true = benchmark.labels[0]
        
            if len(X) > 400:       # max limit size of points
                continue
            
            correct_number_of_clusters = max(y_true)
            
            print("- {0}".format(eachDatasetName))
            print("Dataset size {0}".format(len(X)))

            try:
                figures = []
                figures.append(data_plot.doPCA(
                        X = X, 
                        labels = y_true, 
                        dataset_name = eachDatasetName, 
                        isExample=True
                    )
                )

                # Circle Clustering
                executeClusteringFunction(1, X, eachDatasetName, "CircleClustering", figures, correct_number_of_clusters, y_true)
                # k-means
                executeClusteringFunction(2, X, eachDatasetName, "KMeans", figures, correct_number_of_clusters, y_true)
                # affinity propagation
                executeClusteringFunction(3, X, eachDatasetName, "AffinityPropagation", figures, correct_number_of_clusters, y_true)
                # mean shift
                executeClusteringFunction(4, X, eachDatasetName, "MeanShift", figures, correct_number_of_clusters, y_true)
                # genie
                executeClusteringFunction(5, X, eachDatasetName, "Genie", figures, correct_number_of_clusters, y_true)

                # hierarchical clustering
                # - ward
                # - average linkage
                # - complete linkage
                # - ward linkage

                # dbscan

                # optics

                # birch

                # spectral clustering

                # dbscan

                # optics
                    

                all_figures.append((figures, eachDatasetName))
            except Exception:
                print("Si è verificata un'eccezione : ")
                print(Exception)
                continue

        data_plot.figures_to_html(all_figures)
        break






'''
    battery, dataset = "wut", "x2"
    benchmark = clustbench.load_dataset(battery, dataset, path=data_path)
    X = benchmark.data
    y_true = benchmark.labels[0]
    y_pred = engine.CircleClustering(X)+1

    
    # y_pred[-1] = 3
    for i in range(len(y_true)):
        if y_true[i] == 3:
            y_true[i] = 2
        
    y_pred[-1] = 3

    print(y_true)
    print(y_pred)

    print(len(y_true))
    print(len(y_pred))

    # print(clustbench.get_score(y_true, y_pred))
    print(metrics.adjusted_rand_score(y_true, y_pred))
    
    # engine.CircleClustering(samples, labels, n_dataset)

    

def doClustering(whatClustering, correct_number_of_clusters, X, queue):
    
    switch={
        # 1: (engine.CircleClustering(X) + 1, "CircleClustering"),        # -> returns error
        2: (clustering_sklearn.KMeans(correct_number_of_clusters).fit(X).labels_ + 1, "KMeans"),
        3: (clustering_sklearn.AffinityPropagation().fit(X).labels_ + 1, "Affinity propagation"),
        4: (clustering_sklearn.MeanShift().fit(X).labels_ + 1, "MeanShift"),
        5: (genieclust.Genie(n_clusters=correct_number_of_clusters).fit_predict(X) + 1, "Genie")
        # ...
    }   


    if whatClustering == 2:
        name = "KMeans"
        res = clustering_sklearn.KMeans(correct_number_of_clusters).fit(X).labels_ + 1
    elif whatClustering == 3:
        name = "Affinity Propagation"
        res = clustering_sklearn.AffinityPropagation().fit(X).labels_ + 1
    elif whatClustering == 4:
        name = "Meanshift"
        res = clustering_sklearn.MeanShift().fit(X).labels_ + 1
    elif whatClustering == 5:
        name = "Genie"
        res = genieclust.Genie(n_clusters=correct_number_of_clusters).fit_predict(X) + 1

    print("{0} terminated".format(whatClustering));
    queue.put((name, res))

'''


def executeClusteringFunction(number, X, dataset_name, name, figures, correct_number_of_clusters, y_true):
    if number == 1:
        name = "CircleClustering"
        res = engine.CircleClustering(X) + 1
    elif number == 2:
        name = "KMeans"
        res = clustering_sklearn.KMeans(correct_number_of_clusters).fit(X).labels_ + 1
    elif number == 3:
        name = "Affinity Propagation"
        res = clustering_sklearn.AffinityPropagation().fit(X).labels_ + 1
    elif number == 4:
        name = "Meanshift"
        res = clustering_sklearn.MeanShift().fit(X).labels_ + 1
    elif number == 5:
        name = "Genie"
        res = genieclust.Genie(n_clusters=correct_number_of_clusters).fit_predict(X) + 1

    adjusted_rand_score = metrics.adjusted_rand_score(y_true, res)
    adjusted_mutual_info_score = metrics.adjusted_mutual_info_score(y_true, res)

    figures.append(
        data_plot.doPCA(
            X = X,
            labels = res,
            dataset_name = dataset_name,
            algorithm_name = name, 
            comment =  "Score algorithm : adjusted_rand_score = {1} , adjusted_mutual_info_score = {2}".format(res[1], round(adjusted_rand_score, 4), round(adjusted_mutual_info_score, 4)),
            isExample = False
        )
    )

if __name__ == "__main__":
    main()

