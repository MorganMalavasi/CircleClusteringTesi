addvecs_codetext_kernel = """
__global__ void add_vectors_broadcast(float *dest, float *a, float *b, int* SZ)
{
    const int M = SZ[0];
    const int N = SZ[1];
    const int S = SZ[2];

    const int tx = threadIdx.x;
    const int bx = blockIdx.x;
    const int BSZ = blockDim.x;
    int t;

    for (int s=0;s<S;s++)
    {
        t = s*BSZ+tx;
        if(t<N)
            dest[bx*N+t] = b[t] + a[bx];
        __syncthreads();
    }

}
"""

diagonal_zeros_kernel = """
__global__ void diagonal_zeros(float *mat, int size)
    {
        int idx = threadIdx.x + blockDim.x * blockIdx.x;
        if (idx >= size){
            return;
        }
        
        mat[idx * size + idx] = 0.0;
    }
"""

elementwise_multiplication_kernel = """
__global__ void elementwise_multiplication(float *weights, float *theta, float *C, float*S, float cos, float sin, float valueNewTheta, unsigned int thetaSize, unsigned int k)
    {
        int pos = threadIdx.x + blockDim.x * blockIdx.x; 
        if (pos >= thetaSize)
            return;

        int idx_weight = k * thetaSize + pos;
        C[pos] += (weights[idx_weight] * cos);
        S[pos] += (weights[idx_weight] * sin);
    }    
"""


sklearn_metrics_pairwise_euclidean_distances_kernel = """
__global__ void sklearn_metrics_pairwise_euclidean_distances(float *mat, float *vec, int size)
    {
        int idx = threadIdx.x + blockDim.x * blockIdx.x;
        
    }
"""

dot_prod_gpu_kernel = """
__global__ void dot_product_kernel(float *x, float *y, float *dot, unsigned int n)
{
	unsigned int index = threadIdx.x + blockDim.x * blockIdx.x;
	unsigned int stride = blockDim.x * gridDim.x;

	__shared__ float cache[32];

	double temp = 0.0;
	while(index < n){
		temp += x[index]*y[index];

		index += stride;
	}

	cache[threadIdx.x] = temp;

	__syncthreads();

	// reduction
	unsigned int i = blockDim.x/2;
	while(i != 0){
		if(threadIdx.x < i){
			cache[threadIdx.x] += cache[threadIdx.x + i];
		}
		__syncthreads();
		i /= 2;
	}

	if(threadIdx.x == 0){
		atomicAdd(dot, cache[0]);
	}
}
"""