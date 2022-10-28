import numpy as np
import time
from functools import wraps

def getDataFromGpu(weights, S, C):
    return weights.get(), S.get(), C.get()

def truncate(f, n):
    '''Truncates/pads a float f to n decimal places without rounding'''
    s = '{}'.format(f)
    if 'e' in s or 'E' in s:
        return '{0:.{1}f}'.format(f, n)
    i, p, d = s.partition('.')
    return '.'.join([i, (d+'0'*n)[:n]])

def truncatefloat6digits(n):
    txt = f"(n:.6f)"
    y = float(txt)
    return y
 
def trunc(a, x):
    int1 = int(a * (10**x))/(10**x)
    return float(int1)

####### checker equality between arrays and matrices #######

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

	
def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f'Function {func.__name__} took {total_time:.4f} seconds')
        return result
    return timeit_wrapper