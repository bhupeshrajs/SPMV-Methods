#include "genresult.cuh"
#include <sys/time.h>

__global__ void getMulAtomic_kernel(int nz, int *rIndices, int *cIndices, float *values, int M, int N, float *vector, float *result){
    
    int thread_id = threadIdx.x + (blockIdx.x *blockDim.x);
    int total_number_of_threads = blockDim.x * gridDim.x;
    int iteration = nz % total_number_of_threads ? nz/total_number_of_threads + 1 : nz/total_number_of_threads;
    for( int i = 0 ; i < iteration ; i++ ) {
        
        int data_id = thread_id + i* total_number_of_threads;
        if( data_id < nz ) {
            float data = values[data_id];
            int row = rIndices[data_id];
            int column = cIndices[data_id];
            float multiplication_value = data * vector[column];
            atomicAdd(&result[row],multiplication_value);
        }
    }
}

typedef struct cooFormat {
    int row;
    int column;
    float value;
}cooFormat;

int compareFunction (const void * a, const void * b)
{
    int l = ((cooFormat*)a)->row;
    int r = ((cooFormat*)b)->row;
    return (l-r);
}

void getMulAtomic(MatrixInfo * mat, MatrixInfo * vec, MatrixInfo * res, int blockSize, int blockNum){
    /*Allocate here...*/

    /* Get the values required from the MatrixInfo Structure */
    int number_of_non_zeros = mat->nz;
    int *row_indices = mat->rIndex;
    int *column_indices = mat->cIndex;
    float *values = mat->val;
    int M = mat->M;
    int N = mat->N;
    float *vector = vec->val;
    float *result = res->val;
    
    
    /* Sorting the rows in the order */
    cooFormat *sorting = (cooFormat*)malloc(sizeof(cooFormat)*number_of_non_zeros);
    
    for( int i = 0; i < number_of_non_zeros ; i++ ) {
        sorting[i].row = row_indices[i];
        sorting[i].column = column_indices[i];
        sorting[i].value = values[i];
    }
    
    qsort(sorting,number_of_non_zeros,sizeof(cooFormat),compareFunction);
    
    for( int i = 0; i < number_of_non_zeros ; i++ ) {
        row_indices[i] = sorting[i].row;
        column_indices[i] = sorting[i].column;
        values[i] = sorting[i].value;
    }
    
    printf("\nGPU Code");
    printf("\nBlock Size : %lu, Number of Blocks : %lu, nz : %lu\n",blockSize,blockNum,number_of_non_zeros);
    
    /* Device copies of the required values */
    int   * d_rIndices;
    int   * d_cIndices;
    float *d_values;
    float *d_vector;
    float *d_result;
    
    
    /* Allocate values for the device copies */
    cudaMalloc((void**)&d_rIndices, sizeof(int)*number_of_non_zeros);
    cudaMalloc((void**)&d_cIndices, sizeof(int)*number_of_non_zeros);
    cudaMalloc((void**)&d_values, sizeof(float)*number_of_non_zeros);
    cudaMalloc((void**)&d_vector, sizeof(float)*N);
    cudaMalloc((void**)&d_result, sizeof(float)*M);
    
    /* Set all the result values to be zeros */
    cudaMemset(d_result, 0, sizeof(float)*M);
    
    /* Copying values from host to device */
    cudaMemcpy(d_rIndices,row_indices,sizeof(int)*number_of_non_zeros, cudaMemcpyHostToDevice);
    cudaMemcpy(d_cIndices,column_indices, sizeof(int)*number_of_non_zeros , cudaMemcpyHostToDevice);
    cudaMemcpy(d_values,values, sizeof(float)*number_of_non_zeros,cudaMemcpyHostToDevice);
    cudaMemcpy(d_vector,vector, sizeof(float)*N, cudaMemcpyHostToDevice);
    
		/* Sample timing code */
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC_RAW, &start);
    /*Invoke kernels...*/

    getMulAtomic_kernel<<<blockNum,blockSize>>>(number_of_non_zeros,d_rIndices,d_cIndices,d_values,M,N,d_vector,d_result);
    
    cudaDeviceSynchronize();
    clock_gettime(CLOCK_MONOTONIC_RAW, &end);
    printf("Atomic Kernel Time: %lu milli-seconds\n", 1000 * (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1000000);
    
    cudaMemcpy(result,d_result,sizeof(float)*M, cudaMemcpyDeviceToHost);
    res->val = result;
    
    /*Deallocate.*/
    cudaFree(d_rIndices);
    cudaFree(d_cIndices);
    cudaFree(d_values);
    cudaFree(d_vector);
    cudaFree(d_result);


    /* Occupancy Calculator */
    
    int device;
    cudaDeviceProp prop;
    int activeWarps;
    int maxWarps;
    
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&prop, device);
    
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                                                  &blockNum,
                                                  getMulAtomic_kernel,
                                                  blockSize,
                                                  0);
    
    activeWarps = blockNum * blockSize / prop.warpSize;
    maxWarps = prop.maxThreadsPerMultiProcessor / prop.warpSize;
    
    printf("Occupancy: %f\n",(double)activeWarps / maxWarps * 100);
    printf("Max blocks used by the kernel %d\n",blockNum);


}
