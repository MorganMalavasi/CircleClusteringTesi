import subprocess, os, csv
import numpy as np
from sklearn import metrics
from sklearn.metrics.pairwise import euclidean_distances

def silhouette(samples, labels):
    return metrics.silhouette_score(X=samples, labels=labels, metric="euclidean")

def calinski(samples, labels):
    return metrics.calinski_harabasz_score(X=samples, labels=labels)


def delta(ck, cl):
    values = np.ones([len(ck), len(cl)])*10000
    
    for i in range(0, len(ck)):
        for j in range(0, len(cl)):
            values[i, j] = np.linalg.norm(ck[i]-cl[j])
            
    return np.min(values)
    
def big_delta(ci):
    values = np.zeros([len(ci), len(ci)])
    
    for i in range(0, len(ci)):
        for j in range(0, len(ci)):
            values[i, j] = np.linalg.norm(ci[i]-ci[j])
            
    return np.max(values)
    
def dunn(k_list):
    """ Dunn index [CVI]
    
    Parameters
    ----------
    k_list : list of np.arrays
        A list containing a numpy array for each cluster |c| = number of clusters
        c[K] is np.array([N, p]) (N : number of samples in cluster K, p : sample dimension)
    """
    deltas = np.ones([len(k_list), len(k_list)])*1000000
    big_deltas = np.zeros([len(k_list), 1])
    l_range = list(range(0, len(k_list)))
    
    for k in l_range:
        for l in (l_range[0:k]+l_range[k+1:]):
            deltas[k, l] = delta(k_list[k], k_list[l])
        
        big_deltas[k] = big_delta(k_list[k])

    di = np.min(deltas)/np.max(big_deltas)
    return di

def delta_fast(ck, cl, distances):
    values = distances[np.where(ck)][:, np.where(cl)]
    values = values[np.nonzero(values)]

    return np.min(values)
    
def big_delta_fast(ci, distances):
    values = distances[np.where(ci)][:, np.where(ci)]
    #values = values[np.nonzero(values)]
            
    return np.max(values)

def dunn_fast(points, labels):
    """ Dunn index - FAST (using sklearn pairwise euclidean_distance function)
    
    Parameters
    ----------
    points : np.array
        np.array([N, p]) of all points
    labels: np.array
        np.array([N]) labels of all points
    """
    distances = euclidean_distances(points)
    ks = np.sort(np.unique(labels))
    
    deltas = np.ones([len(ks), len(ks)])*1000000
    big_deltas = np.zeros([len(ks), 1])
    
    l_range = list(range(0, len(ks)))
    
    for k in l_range:
        for l in (l_range[0:k]+l_range[k+1:]):
            deltas[k, l] = delta_fast((labels == ks[k]), (labels == ks[l]), distances)
        
        big_deltas[k] = big_delta_fast((labels == ks[k]), distances)

    di = np.min(deltas)/np.max(big_deltas)
    return di
    
    
def  big_s(x, center):
    len_x = len(x)
    total = 0
        
    for i in range(len_x):
        total += np.linalg.norm(x[i]-center)    
    
    return total/len_x

def davisbouldin(k_list, k_centers):
    """ Davis Bouldin Index
    
    Parameters
    ----------
    k_list : list of np.arrays
        A list containing a numpy array for each cluster |c| = number of clusters
        c[K] is np.array([N, p]) (N : number of samples in cluster K, p : sample dimension)
    k_centers : np.array
        The array of the cluster centers (prototypes) of type np.array([K, p])
    """
    len_k_list = len(k_list)
    big_ss = np.zeros([len_k_list], dtype=np.float64)
    d_eucs = np.zeros([len_k_list, len_k_list], dtype=np.float64)
    db = 0    

    for k in range(len_k_list):
        big_ss[k] = big_s(k_list[k], k_centers[k])

    for k in range(len_k_list):
        for l in range(0, len_k_list):
            d_eucs[k, l] = np.linalg.norm(k_centers[k]-k_centers[l])

    for k in range(len_k_list):
        values = np.zeros([len_k_list-1], dtype=np.float64)
        for l in range(0, k):
            values[l] = (big_ss[k] + big_ss[l])/d_eucs[k, l]
        for l in range(k+1, len_k_list):
            values[l-1] = (big_ss[k] + big_ss[l])/d_eucs[k, l]

        db += np.max(values)
    res = db/len_k_list
    return res


def widest_within_cluster_gap_formula(samples, labels):
    # Defining the R script and loading the instance in Python
    createFile(samples, labels)
    score = command_wwcg()
    deleteFile()        
    return score

def pearson(samples, labels):
    createFile(samples, labels)
    score = command_pearson()
    deleteFile()
    return score

def createFile(samples, labels, cvnn = False, labels2 = None):    
    path_to_file_samples = 'analysis/cqcluster/k_means_input.csv'
    
    # printMatrix(samples, labels)
    
    with open(path_to_file_samples, "w") as csvfile:
        csvwriter = csv.writer(csvfile)
        
        for i in range (samples.shape[0]):
            if i == 0:
                size = samples.shape[1]
                write_col_of_Data_frame(csvwriter, size)
            
            row = samples[i,:]
            csvwriter.writerow(row)

    path_to_file_labels = 'analysis/cqcluster/labels_input.csv'

    with open(path_to_file_labels, "w") as csvfile:
        csvwriter = csv.writer(csvfile)

        for i in range(labels.shape[0]):
            if i == 0:
                size = 1
                write_col_of_Data_frame(csvwriter, size)
            
            row = [labels[i] + 1]
            csvwriter.writerow(row)

    if cvnn:
        path_to_file_labels2 = 'analysis/cqcluster/labels_input_2.csv'

        with open(path_to_file_labels2, "w") as csvfile:
            csvwriter = csv.writer(csvfile)

            for i in range(labels2.shape[0]):
                if i == 0:
                    size = 1
                    write_col_of_Data_frame(csvwriter, size)
                
                row = [int(labels2[i]) + 1]
                csvwriter.writerow(row)
    return 

def write_col_of_Data_frame(csvwriter, size):
    myTuple = []
    for k in range(size):
        myTuple.append(str(k)) 
        
    # print(myTuple)
                
    csvwriter.writerow(myTuple)

def deleteFile():
    if os.path.exists("analysis/cqcluster/k_means_input.csv"):
        os.remove("analysis/cqcluster/k_means_input.csv")
    if os.path.exists("analysis/cqcluster/labels_input.csv"):
        os.remove("analysis/cqcluster/labels_input.csv")
    if os.path.exists("analysis/cqcluster/labels_input_2.csv"):
        os.remove("analysis/cqcluster/labels_input_2.csv")
    return 

def command_wwcg():
    command = 'Rscript'
    # command = 'Rscript'                    # OR WITH bin FOLDER IN PATH ENV VAR 
    arg = '--vanilla' 

    try: 
        p = subprocess.Popen([command, arg,
                            "analysis/cqcluster/widest_within_cluster_gap.R"],
                            cwd = os.getcwd(),
                            stdin = subprocess.PIPE, 
                            stdout = subprocess.PIPE, 
                            stderr = subprocess.PIPE) 

        output, error = p.communicate() 

        if p.returncode == 0: 
            # print('R OUTPUT:\n {0}'.format(output.decode("utf-8"))) 
            out = output.decode("utf-8")
            out = out.replace('[1]', '')
            return float(out)
        else: 
            print('R ERROR:\n {0}'.format(error.decode("utf-8"))) 
            return None

    except Exception as e: 
        print("dbc2csv - Error converting file: ") 
        print(e)

        return False

def command_pearson():
    command = 'Rscript'
    # command = 'Rscript'                    # OR WITH bin FOLDER IN PATH ENV VAR 
    arg = '--vanilla' 

    try: 
        p = subprocess.Popen([command, arg,
                            "analysis/cqcluster/pearson.R"],
                            cwd = os.getcwd(),
                            stdin = subprocess.PIPE, 
                            stdout = subprocess.PIPE, 
                            stderr = subprocess.PIPE) 

        output, error = p.communicate() 

        if p.returncode == 0: 
            # print('R OUTPUT:\n {0}'.format(output.decode("utf-8"))) 
            out = output.decode("utf-8")
            out = out.replace('[1]', '')
            return float(out)
        else: 
            print('R ERROR:\n {0}'.format(error.decode("utf-8"))) 
            return None

    except Exception as e: 
        print("dbc2csv - Error converting file: ") 
        print(e)

        return False