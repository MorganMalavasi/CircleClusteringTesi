from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs
from sklearn.datasets import make_moons, make_circles, make_classification, fetch_olivetti_faces
from data_plot import doPCA

def scaling(samples):
    scaler = StandardScaler()
    samples = scaler.fit_transform(samples)
    return samples

def create_dataset_base(samples, features, centers, standard_deviation_cluster = 1, standard = True, display = False, n_dataset = 0):
    # X = The generated samples
    # l = The integer labels for cluster membership of each sample
    X, l = make_blobs(n_samples = samples, n_features = features, centers = centers, cluster_std = standard_deviation_cluster, random_state = None)
    if standard:
        X = scaling(X)
    if display: 
        doPCA(X, l, n_dataset)
    return X, l, n_dataset

def create_dataset_moon(samples, noise, standard = True, display = False, n_dataset = 0):
    X, l = make_moons(n_samples = samples, noise = noise)
    if standard:
        X = scaling(X)
    if display:
        doPCA(X, l, n_dataset)
    return X, l, n_dataset

def create_dataset_circles(samples, noise, standard = True, display = False, n_dataset = 0):
    X, l = make_circles(n_samples = samples, noise = noise)
    if standard:
        X = scaling(X)
    if display:
        doPCA(X, l, n_dataset)
    return X, l, n_dataset

def create_dataset_classification(n_samples, n_features, n_redundant, n_informative, n_clustes_per_class, display = False, n_dataset = 0, standard = True):
    X, l = make_classification(n_samples = n_samples, n_features = n_features, n_redundant = n_redundant, n_informative = n_informative, n_clusters_per_class=n_clustes_per_class)
    if standard:
        X = scaling(X)
    if display:
        doPCA(X, l, n_dataset)
    return X, l, n_dataset

def create_dataset_olivetti_faces(display = False, n_dataset = 0, standard = True):
    data = fetch_olivetti_faces()
    X = data.data
    l = data.target
    if standard:
        X = scaling(X)
    if display:
        doPCA(X, l, n_dataset)
    return X, l, n_dataset


def createDatasets(key = None):
    '''
    sample0, l0, n_dataset0 = create_dataset_base(samples = 20, features = 5, centers = 3, display = False, n_dataset = 0)
    sample1, l1, n_dataset1 = create_dataset_base(samples = 1000, features = 5, centers = 5, display = False, n_dataset = 1)
    sample2, l2, n_dataset2 = create_dataset_base(samples = 4000, features = 7, centers = 10, display = False, n_dataset = 2)
    sample3, l3, n_dataset3 = create_dataset_base(samples = 7000, features = 15, centers = 8, standard_deviation_cluster=1.5, display = False, n_dataset = 3)
    sample4, l4, n_dataset4 = create_dataset_base(samples = 10000, features = 1024, centers = 5, display = False, n_dataset = 4)
    sample5, l5, n_dataset5 = create_dataset_base(samples = 13000, features = 1024, centers = 5, display = False, n_dataset = 5)
    sample6, l6, n_dataset6 = create_dataset_base(samples = 15000, features = 2048, centers = 5, display = False, n_dataset = 6)
    sample7, l7, n_dataset7 = create_dataset_base(samples = 17000, features = 2048, centers = 8, display = False, n_dataset = 7)
    sample8, l8, n_dataset8 = create_dataset_moon(samples = 1000, noise = 0.05, display = False, n_dataset = 8)            
    sample9, l9, n_dataset9 = create_dataset_moon(samples = 1000, noise = 0.2, display = False, n_dataset = 9)
    sample10, l10, n_dataset10 = create_dataset_circles(samples = 1000, noise = 0.05, display = False, n_dataset = 10)
    sample11, l11, n_dataset11 = create_dataset_classification(n_samples = 100, n_features = 2, n_redundant = 0, n_informative = 2, n_clustes_per_class=1, display = False, n_dataset = 11)
    sample12, l12, n_dataset12 = create_dataset_classification(n_samples = 1000, n_features = 2, n_redundant = 0, n_informative = 2, n_clustes_per_class=1, display = False, n_dataset = 12)
    sample13, l13, n_dataset13 = create_dataset_classification(n_samples = 5000, n_features = 8, n_redundant = 0, n_informative = 2, n_clustes_per_class=1, display = False, n_dataset = 13)
    sample14, l14, n_dataset14 = create_dataset_olivetti_faces(display = False, n_dataset = 14)
    # sample15, l15, n_dataset15 = create_dataset_base(samples = 50000, features = 2048, centers = 10, display = False, n_dataset = 15)

    listOfDataset = []

    listOfDataset.append((sample0, l0, n_dataset0))
    listOfDataset.append((sample1, l1, n_dataset1))
    listOfDataset.append((sample2, l2, n_dataset2))
    listOfDataset.append((sample3, l3, n_dataset3))
    listOfDataset.append((sample4, l4, n_dataset4))
    listOfDataset.append((sample5, l5, n_dataset5))
    listOfDataset.append((sample6, l6, n_dataset6))
    listOfDataset.append((sample7, l7, n_dataset7))
    listOfDataset.append((sample8, l8, n_dataset8))
    listOfDataset.append((sample9, l9, n_dataset9))
    listOfDataset.append((sample10, l10, n_dataset10))
    listOfDataset.append((sample11, l11, n_dataset11))
    listOfDataset.append((sample12, l12, n_dataset12))
    listOfDataset.append((sample13, l13, n_dataset13))
    listOfDataset.append((sample14, l14, n_dataset14))
    ''' 

    switcher = {
        0 : create_dataset_base(samples = 500, features = 2, centers = 3, display = False, n_dataset = 0),
        1 : create_dataset_base(samples = 800, features = 10, centers = 4, display = False, n_dataset = 1),
        2 : create_dataset_base(samples = 5000, features = 6, centers = 6, standard_deviation_cluster=1.2, display = False, n_dataset = 2),
        3 : create_dataset_classification(n_samples = 500, n_features = 3, n_redundant = 0, n_informative = 2, n_clustes_per_class=1, display = False, n_dataset = 3),
        4 : create_dataset_circles(samples = 1000, noise = 0.05, display = False, n_dataset = 4),
        5 : create_dataset_moon(samples = 1000, noise = 0.05, display = False, n_dataset = 5),
        6 : create_dataset_base(samples = 7000, features = 15, centers = 8, standard_deviation_cluster=1.5, display = False, n_dataset = 6)
    }

    return switcher.get(key, "Invalid Input")