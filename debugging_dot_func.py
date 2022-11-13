import cclustering.cclustering_cpu as c_cpu
import cclustering.cclustering_gpu as c_gpu
import cclustering.utils as utils
import numpy as np
import sklearn.datasets as db

# constants
PI = np.pi
PI = np.float32(PI)

samples = np.array([[0.000000e+000, 0.000000e+000], [1.000000e-001, -1.000000e-001], [2.000000e-001, -2.000000e-001]])

numberOfSamplesInTheDataset = samples.shape[0]
theta = 2 * PI * np.random.rand(numberOfSamplesInTheDataset)

theta_method1 = np.copy(theta)
theta_method2 = np.copy(theta)
theta_method3 = np.copy(theta)

matrixOfWeights, S, C = c_cpu.computing_weights(samples, theta_method1, cosine = False)
Snew, Cnew = c_cpu.computing_weights_memory_aware(samples, theta_method2, cosine = False)
S_gpu, C_gpu = c_gpu.computing_S_C_memory_aware_gpu(samples, theta_method3)

print(utils.checkResultsMatrixWeights(S_gpu, Snew, 0.0001))
