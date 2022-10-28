import numpy as np
from sklearn.metrics.pairwise import euclidean_distances, cosine_distances
from numpy import linalg
from numba import jit

'''
    If the number of samples remains low, use the tradition way, it is faster
'''
def computing_weights(dataset, theta, cosine = False):
    if cosine == False:
        weights = euclidean_distances(dataset, dataset)
    else :
        weights = cosine_distances(dataset, dataset)
    
    weights = weights / linalg.norm(weights)

    S, C = C_S(weights, theta)
    return weights, S, C

@jit
def C_S(matrixOfWeights, theta):
    sin_t = np.sin(theta)
    S = np.dot(matrixOfWeights, sin_t)

    cos_t = np.cos(theta)
    C = np.dot(matrixOfWeights, cos_t)
    
    return S, C

@jit(nopython=True) 
def loop(matrixOfWeights, theta, S, C, eps):

    PI = np.pi
    PI = np.float32(PI)
    
    ok = True
    rounds = 0
    thetaSize = theta.shape[0]

    while ok == True:
        ok = False
        rounds += 1
        
        ''' loop on the theta '''
        for k in range(thetaSize):

            old = theta[k]
            
            ''' find a theta that improves the equation '''
            theta[k] = np.arctan(S[k]/C[k])

            if C[k] >= 0:
                theta[k] += PI
            elif S[k] > 0:
                theta[k] += 2*PI
                
            jit_elementwise_multiplication(matrixOfWeights[k,:], C, S, theta, k, old)

            ''' exit condition '''
            if min(abs(old - theta[k]), abs(2*PI - old + theta[k])) > eps:
                ok = True
    
    return theta

@jit(nopython=True, parallel=True)
def jit_elementwise_multiplication(line_weights, C, S, theta, k, old):
    # elementwise multiplication
    C += np.multiply(line_weights, np.repeat(np.cos(theta[k]) - np.cos(old), theta.shape[0]))
    S += np.multiply(line_weights, np.repeat(np.sin(theta[k]) - np.sin(old), theta.shape[0]))


'''
    If the number of samples start to increase, we have a problem with the memory, we end up with 
    a crash because the machine is 8 or maybe 16 GB, and the dissimilarity matrix exceeds the memory avaiable
'''

def computing_weights_memory_aware(dataset, theta, cosine = False):
    S, C = C_S_memory_aware(dataset, theta)
    return S, C

# @jit
def C_S_memory_aware(dataset, theta):
    sin_t = np.sin(theta)
    weights = euclidean_distances(dataset, dataset)
    
    
    for row in range(dataset.shape[0]):
        l_weights = line_weights(dataset, row)
        # print(areEqual(l_weights, weights[row], 0.000001))
    

    
    '''
    S = np.dot(weights, sin_t)      # -> original without linalg.norm
    
    Snew = np.empty(theta.shape[0])
    for i in range(Snew.shape[0]):
        # compute line of weights
        S[i] = np.dot(line_weights(dataset, i), sin_t)

    print(S)
    print(Snew)
    print((S == Snew).all())
    '''
    
    return None, None

def line_weights(dataset, row):
    line = np.empty(dataset.shape[0])
    for i in range(dataset.shape[0]):
        line[i] = dist(dataset[row], dataset[i])
    return line


def dist(x, y):
    return np.sqrt(np.dot(x, x) - 2 * np.dot(x, y) + np.dot(y, y))

def areEqual(arr1, arr2, error):
    N = arr1.shape[0]
    M = arr2.shape[0]
    # If lengths of array are not
    # equal means array are not equal
    if (N != M):
        print("Different sizes")
        return False
 
    # Linearly compare elements
    for i in range(0, N):
        if abs(abs(arr1[i]) - abs(arr2[i])) > error :
            print(abs(abs(arr1[i]) - abs(arr2[i])))
            print("index = i:{0}".format(i))
            return False
 
    # If all elements were same.
    return True

def areEqualPrecise(arr1, arr2):
    N = arr1.shape[0]
    M = arr2.shape[0]
    # If lengths of array are not
    # equal means array are not equal
    if (N != M):
        print("Different sizes")
        return False
 
    # Linearly compare elements
    for i in range(0, N):
        if (arr1[i] != arr2[i]):
            print("Position = {0}".format(i))
            print(arr1[i])
            print(arr2[i])
            return False
 
    # If all elements were same.
    return True

def checkResultsMatrixWeights(array1, array2, error):
    '''
    if array1.shape[0] != array2.shape[0]:
        return False
    '''
    sizeMatrix = array1.shape[0]
    for i in range(sizeMatrix):
        for j in range(sizeMatrix):
            if abs(abs(array1[i, j]) - abs(array2[i, j])) > error :
                print(abs(abs(array1[i, j]) - abs(array2[i, j])))
                print("index = i:{0} j:{1}".format(i,j))
                return False
    return True

def checkResultsMatrixPrecise(matrix1, matrix2):
    row = matrix1.shape[0]
    col = matrix2.shape[1]
    for i in range(row):
        for j in range(col):
            if matrix1[i, j] != matrix2[i, j]:
                print("difference == 1) {0}    2) {1}".format(matrix1[i, j], matrix2[i, j]))
                return False
    return True