import math
import numpy as np
import skcuda.linalg as culinalg
import pycuda.autoinit
import pycuda.driver as drv
import pycuda.gpuarray as gpuarray
from pycuda.cumath import sqrt as cusqrt
from operator import mul
import more_itertools as mit
import cclustering.utils as utils
import cclustering.cclustering_cpu as cclustering_cpu
import warnings
warnings.filterwarnings("ignore")

culinalg.init()

from pycuda.compiler import SourceModule as SM
from cclustering.cuda_kernels import addvecs_codetext_kernel, diagonal_zeros_kernel, elementwise_multiplication_kernel, dot_prod_gpu_kernel, line_of_weights_gpu_kernel

addvecs_bcast_gpu = SM(addvecs_codetext_kernel).get_function("add_vectors_broadcast")
diagonal_zeros_gpu = SM(diagonal_zeros_kernel).get_function("diagonal_zeros")
elementwise_multiplication = SM(elementwise_multiplication_kernel).get_function("elementwise_multiplication")
dot_product_gpu = SM(dot_prod_gpu_kernel).get_function("dot_product_kernel")
line_of_weights_gpu = SM(line_of_weights_gpu_kernel).get_function("line_of_weights_kernel")

def computing_weights(dataset, theta):
    matrixOfWeights = sqsum_adddot(dataset, dataset)
    
    # set the diagonal of the matrix to zero 
    block = (32, 1, 1)
    grid_x = (matrixOfWeights.shape[0] + (block[0] - 1)) // block[0]
    grid_y = 1
    grid = (grid_x, grid_y)
    size = np.int32(matrixOfWeights.shape[0])
    diagonal_zeros_gpu(matrixOfWeights, size, block = block, grid = grid)

    # sqrt of the values in the matrix
    matrixOfWeights = cusqrt(matrixOfWeights)

    # norm of the matrix
    val = culinalg.norm(matrixOfWeights)
    _weights_ = matrixOfWeights / val

    theta_sin = np.asarray(np.sin(theta) , np.float32)
    theta_cos = np.asarray(np.cos(theta) , np.float32)

    theta_sin_gpu = gpuarray.to_gpu(theta_sin)
    theta_cos_gpu = gpuarray.to_gpu(theta_cos)

    S_GPU = culinalg.dot(_weights_, theta_sin_gpu)
    C_GPU = culinalg.dot(_weights_, theta_cos_gpu)  

    return _weights_, S_GPU, C_GPU


def sqsum_adddot(a,b):
    """
    Compute squared euclidean distance between two 2D arrays representing
    n-dimensional points using GPU. This uses the input arrays themselves to
    compute element-wise summations of squared sum of rows and accumulates into
    the matrix-multiplication result residing on GPU.
    The final result resides on GPU.

    Parameters
    ----------
    A : ndarray
        2D NumPy array of float dtype representing n-dimensional points, with
        each row being one point.
    B : ndarray
        2D NumPy array of float dtype representing n-dimensional points, with
        each row being one point.

    Returns
    -------
    out : GPUArray
        This holds the euclidean distances residing on GPU.
    """
    a = convert_f32(a)
    b = convert_f32(b)

    a_gpu = gpuarray.to_gpu(a)
    b_gpu = gpuarray.to_gpu(b)
    c_gpu = squared_sum(a,b)
    # culinalg.add_dot      C = alpha * (A B) + beta * C
    # transb (char) – If ‘T’, compute the product of the transpose of b_gpu.
    return culinalg.add_dot(a_gpu, b_gpu, c_gpu,  transb='T', alpha=-2.0)

def squared_sum(a,b):
    """
    Compute squared summations of rows and then their pairwise summations.

    Parameters
    ----------
    A : ndarray
    B : ndarray

    Returns
    -------
    out : GPUArray
        Compute squared summations of each row for each of the ndarrays giving us
        two 1D arrays. Then, compute their pairwise summations to result in a 2D
        array.

        Compute squared sum of rows of the inputs, giving us
        two `1D` arrays. Transfer these as two arrays onto GPU. Create a `zeros`
        array directly on GPU and in two steps add in the two summed arrays in a
        broadcasted manner, using own kernel along the rows and
        columns, giving us the pairwise summations.      

    """

    c_gpu = None # Initialize output
    """
        Using the Einstein summation convention, many common multi-dimensional, 
        linear algebraic array operations can be represented in a simple fashion. 
    """
    c_gpu = addvecs(np.einsum('ij,ij->i',a,a), np.einsum('ij,ij->i',b,b))
    return c_gpu

def addvecs(a, b, output='gpu'):
    """
    Add two 1D arrays for all pairs of elements resulting in a 2D array.

    Parameters
    ----------
    A : ndarray or GPUArray
    B : ndarray or GPUArray
    output : str, optional
        Selects the output datatype. It can be 'cpu' or 'gpu'.

    Returns
    -------
    out : ndarray or GPUArray
        Pairwise summation of elements from input 1D arrays. Thus, if first
        array has M elements and second one has N elements, we would have an
        output array of shape (M,N). The output class would be GPUArray or
        ndarray class, depending on the input argument 'output'. This decides
        whether the final output is to be kept on the GPU or brought back to
        the CPU host respectively.

    """

    if str(type(a)).find('gpuarray')!=-1:
        a_gpu = a
    elif str(type(b)).find('ndarray')!=-1:
        a_gpu = gpuarray.to_gpu(a)
    else:
        raise Exception("Input type invalid")

    if str(type(b)).find('gpuarray')!=-1:
        b_gpu = b
    elif str(type(b)).find('ndarray')!=-1:
        b_gpu = gpuarray.to_gpu(b)
    else:
        raise Exception("Input type invalid")

    M, N = a_gpu.shape[0], b_gpu.shape[0]
    out_gpu = gpuarray.empty((M,N),dtype=np.float32)
    BSZ = min(1024,N)
    GSZ = M
    num_iter = int(np.ceil(N/float(1024)))
    a_shp = np.int32([M,N,num_iter])
    addvecs_bcast_gpu(out_gpu, a_gpu, b_gpu, drv.In(a_shp), block=(BSZ,1,1), grid=(GSZ,1))

    if output=='gpu':
        return out_gpu
    elif output=='cpu':
        return out_gpu.get()
    else:
        raise Exception("Output type invalid")
    

def loop_gpu(weights, theta, S, C, eps):

    PI = np.pi
    PI = np.float32(PI)
    
    theta = gpuarray.to_gpu(theta)
    block = (32, 1, 1)
    grid_x = (weights.shape[1] + (block[0] - 1)) // block[0]
    grid_y = 1
    grid = (grid_x, grid_y)

    # preparing kernel
    kernel_call = elementwise_multiplication.prepare("PPPPfffii").prepared_call

    ok = True
    rounds = 0 
    thetaSize = theta.shape[0]
    thetaSize_int32 = np.uint32(thetaSize)

    while ok == True:
        ok = False
        rounds += 1

        for k in range(thetaSize):
            old = theta[k].get().item()
            sin = S[k].get().item()
            cos = C[k].get().item()
            
            tmp = math.atan(sin/cos)
            
            if cos >= 0:
                tmp += PI
            elif sin > 0:
                tmp += 2*PI

            tmp_ = np.array([tmp]) 
            theta[k].set(tmp_)
            
            val_cos = math.cos(tmp) - math.cos(old)
            val_sin = math.sin(tmp) - math.sin(old)

            val_cos_float32 = np.float32(val_cos)
            val_sin_float32 = np.float32(val_sin)
            val_new_theta = np.float32(tmp)
            k_int32 = np.uint32(k)
           
            kernel_call(grid, block, weights.gpudata, theta.gpudata, C.gpudata, S.gpudata, val_cos_float32, val_sin_float32, val_new_theta, thetaSize, k)
            
            if min(abs(old - tmp), abs(2*PI - old + tmp)) > eps:
                ok = True

    return theta.get()

@utils.timeit
def computing_S_C_memory_aware_gpu(samples, theta, cosine = False):
    sin_t = np.sin(theta)
    cos_t = np.cos(theta)

    '''
    Snew = np.empty(theta.shape[0])
    Cnew = np.empty(theta.shape[0])
    '''
    Snew = np.empty((theta.shape[0], theta.shape[0]))
    Cnew = np.empty((theta.shape[0], theta.shape[0]))

    dataset = convert_f32(samples)
    dataset_flat = np.asarray(dataset).ravel()      # important!!! the matrix needs to be flatted or will be rotated in the kernel
    dataset_gpu = gpuarray.to_gpu(dataset_flat)
    

    # compute line of weights
    block = (32, 1, 1)
    grid_x = (theta.shape[0] +(block[0] - 1)) // block[0]
    grid_y = 1
    grid = (grid_x, grid_y)
    size = np.int32(theta.shape[0])
    features = np.int32(samples.shape[1])

    for i in range(theta.shape[0]):
        i_32 = np.int32(i)

        line_weights = np.zeros(theta.shape[0])
        line_weights = convert_f32(line_weights)
        line_weights_gpu = gpuarray.to_gpu(line_weights)

        line_of_weights_gpu(dataset_gpu, i_32, line_weights_gpu, size, features, block = block, grid = grid)
        line_weights = line_weights_gpu.get()
        # if i == 0:
        # print(line_weights)

        # Snew[i] = np.dot(line_weights, sin_t)
        # Cnew[i] = np.dot(line_weights, cos_t)
        Snew[i] = line_weights
        Cnew[i] = line_weights

    return Snew, Cnew
    
@utils.timeit
def loop_memory_aware_gpu(dataset, theta, S, C, eps):

    PI = np.pi
    PI = np.float32(PI)
    
    ok = True
    rounds = 0
    thetaSize = theta.shape[0]


    dataset_f32 = convert_f32(dataset)
    dataset_flat = np.asarray(dataset_f32).ravel()      # important!!! the matrix needs to be flatted or will be rotated in the kernel
    dataset_gpu = gpuarray.to_gpu(dataset_flat)

    # compute line of weights
    block = (32, 1, 1)
    grid_x = (theta.shape[0] +(block[0] - 1)) // block[0]
    grid_y = 1
    grid = (grid_x, grid_y)
    size = np.int32(theta.shape[0])
    features = np.int32(dataset.shape[1])


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
            
            # compute line of weights
            k_32 = np.int32(k)

            line_weights = np.zeros(theta.shape[0])
            line_weights = convert_f32(line_weights)
            line_weights_gpu = gpuarray.to_gpu(line_weights)

            line_of_weights_gpu(dataset_gpu, k_32, line_weights_gpu, size, features, block = block, grid = grid)
            line_weights = line_weights_gpu.get()
            
            cclustering_cpu.jit_elementwise_multiplication(line_weights, C, S, theta, k, old)

            ''' exit condition '''
            if min(abs(old - theta[k]), abs(2*PI - old + theta[k])) > eps:
                ok = True
    
    return theta


''' dot product functions '''
@utils.timeit
def dot_prod_gpu(a, b, size, blockSize):
    c = np.zeros(size)
    c = convert_f32(c)

    a_gpu = gpuarray.to_gpu(a)
    b_gpu = gpuarray.to_gpu(b)
    c_gpu = gpuarray.to_gpu(c)

    block = (blockSize, 1, 1)
    grid_x = (a_gpu.shape[0] + (block[0] - 1)) // block[0]
    grid_y = 1
    grid = (grid_x, grid_y)
    dim = np.int32(size)

    dot_product_gpu(a_gpu, b_gpu, c_gpu, dim, block = block, grid = grid)
    return c_gpu.get()


# dot product with numpy
@utils.timeit
def dot_prod_cpu_numpy(x1, x2):
    return np.dot(x1, x2)

# dot product with sum
@utils.timeit
def dot_prod_cpu_sum(a, b):
    return sum([i*j for (i, j) in zip(a, b)])

# dot product with map
@utils.timeit
def dot_prod_cpu_map(a, b):
    return sum(map(mul, a, b))

# dot product with mit
@utils.timeit
def dot_prod_cpu_mit(a, b):
    return mit.dotproduct(a, b)




def getData(weights, S, C):
    return weights.get(), S.get(), C.get()

def convert_f32(a):
    """
    Convert to float32 dtype.

    Parameters
    ----------
    a : ndarray

    Returns
    -------
    out : ndarray
        Converts to float32 dtype if not already so. This is needed for
        implementations that work exclusively work such datatype.

    """

    if a.dtype!=np.float32:
        return a.astype(np.float32)
    else:
        return a

def convert_f64(a):
    """
    Convert to float64 dtype.

    Parameters
    ----------
    a : ndarray

    Returns
    -------
    out : ndarray
        Converts to float64 dtype if not already so. This is needed for
        implementations that work exclusively work such datatype.

    """

    if a.dtype!=np.float64:
        return a.astype(np.float64)
    else:
        return a