from turtle import pos
import numpy as np
import sys

def smooth(values, smoothing_index = 10):
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
    for i in range(values.shape[0]):
        sum = 0.0
        count = 0
        for j in range(smoothing_index):
            x = j - int(smoothing_index/2)
            if (i+x)>=0 and (i+x)<values.shape[0]:
                sum = sum + values[i+x]
                count += 1

        output[i] = sum / count

    return output

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

def simple_detection(values):   
    weights = np.zeros([values.shape[0]])
    values[0] = 10000
    values[values.shape[0]-1] = 10000

    for i in range(1, values.shape[0] - 1):
        falling(values, weights, i)

    count = 0
    for i in range(weights.shape[0]):
        if weights[i] > 0:
            count += 1

    return count - 1, weights


def falling(values, weights, position):
    left = values[position-1]
    right = values[position+1]
    
    # the values are equal and the same -> go the left 
    if left == values[position] and right == values[position]:
        falling(values, weights, position - 1)
    # found a hole -> stop
    elif left > values[position] and right > values[position]:
        weights[position] += 1
    # or the side values are smaller -> go to the smaller one, gravity is major
    elif left < values[position] and right < values[position]:
        if left <= right:
            falling(values, weights, position - 1)
        else:
            falling(values, weights, position + 1)
    # or the left value is smaller or equal and the right value is bigger -> go to the left
    elif left <= values[position] and right > values[position]:
        falling(values, weights, position - 1)
    # or the left value is bigger and the right value is smaller -> go to the right
    elif left > values[position] and right < values[position]:
        falling(values, weights, position + 1)
    # or the left value is bigger and the right value is equal -> stop
    elif left > values[position] and right == values[position]:
        weights[position] += 1
    
    

    