#include "genresult.cuh"
#include <sys/time.h>

/* Put your own kernel(s) here*/
__device__ float segmentedScan_32_WarpSize( int lane , int * rows , float *values ) {
    lane = lane % 32;
    if( lane >= 1 ) {
        values[threadIdx.x] += values[threadIdx.x - 1];
    }
    if( lane >= 2 ) {
        values[threadIdx.x] += values[threadIdx.x - 2];
    }
    if( lane >= 4 ) {
        values[threadIdx.x] += values[threadIdx.x - 4];
    }
    if( lane >= 8 ) {
        values[threadIdx.x] += values[threadIdx.x - 8];
    }
    if( lane >= 16 ) {
        values[threadIdx.x] += values[threadIdx.x - 16];
    }
    return values[threadIdx.x];
}

__device__ float segmentedScan_16_WarpSize( int lane , int * rows , float *values ) {
    lane = lane % 16;
    if( lane >= 1 ) {
        values[threadIdx.x] += values[threadIdx.x - 1];
    }
    if( lane >= 2 ) {
        values[threadIdx.x] += values[threadIdx.x - 2];
    }
    if( lane >= 4 ) {
        values[threadIdx.x] += values[threadIdx.x - 4];
    }
    if( lane >= 8 ) {
        values[threadIdx.x] += values[threadIdx.x - 8];
    }
    return values[threadIdx.x];
}

__device__ float segmentedScan_8_WarpSize( int lane , int * rows , float *values ) {
    lane = lane % 8;
    if( lane >= 1 ) {
        values[threadIdx.x] += values[threadIdx.x - 1];
    }
    if( lane >= 2 ) {
        values[threadIdx.x] += values[threadIdx.x - 2];
    }
    if( lane >= 4 ) {
        values[threadIdx.x] += values[threadIdx.x - 4];
    }
    return values[threadIdx.x];
}

__device__ float segmentedScan_4_WarpSize( int lane , int * rows , float *values ) {
    lane = lane % 4;
    if( lane >= 1 ) {
        values[threadIdx.x] += values[threadIdx.x - 1];
    }
    if( lane >= 2 ) {
        values[threadIdx.x] += values[threadIdx.x - 2];
    }
    return values[threadIdx.x];
}

__device__ float segmentedScan_2_WarpSize( int lane , int * rows , float *values ) {
    lane = lane % 2;
    if( lane >= 1 ) {
        values[threadIdx.x] += values[threadIdx.x - 1];
    }
    return values[threadIdx.x];
}

__global__ void workLoad32Size(int nz, int offset,  int *rIndices, int *cIndices, float *values, int M, int N, float *vector, float *result ) {
    
    int thread_id = threadIdx.x + (blockIdx.x *blockDim.x);
    int total_number_of_threads = blockDim.x * gridDim.x;
    int iteration = nz % total_number_of_threads ? nz/total_number_of_threads + 1 : nz/total_number_of_threads;
    
    __shared__ int rows[1024];
    __shared__ float vals[1024];
    
    int warpId = threadIdx.x >> 5;
    int warpFirst = warpId << 5;
    int warpLast = warpFirst + 31;
    for( int i = 0 ; i < iteration ; i++ ) {
        int data_id = thread_id + i* total_number_of_threads + offset;
        if( data_id < nz ) {
            float data = values[data_id];
            int row = rIndices[data_id];
            int column = cIndices[data_id];
            float multiplication_value = data * vector[column];
            vals[threadIdx.x] = multiplication_value;
            rows[threadIdx.x] = row;
            __syncthreads();
            int lane = threadIdx.x % 32;
            float val = segmentedScan_32_WarpSize(lane,rows,vals);
            __syncthreads();
            if( lane % 32 == 31 ) {
                atomicAdd(&result[rows[threadIdx.x]],vals[threadIdx.x]);
            }
            __syncthreads();
        }
    }
}

__global__ void workLoad16Size(int nz, int offset,  int *rIndices, int *cIndices, float *values, int M, int N, float *vector, float *result ) {
    
    int thread_id = threadIdx.x + (blockIdx.x *blockDim.x);
    int total_number_of_threads = blockDim.x * gridDim.x;
    int iteration = nz % total_number_of_threads ? nz/total_number_of_threads + 1 : nz/total_number_of_threads;
    
    __shared__ int rows[1024];
    __shared__ float vals[1024];
    
    int warpId = threadIdx.x >> 5;
    int warpFirst = warpId << 5;
    int warpLast = warpFirst + 31;
    for( int i = 0 ; i < iteration ; i++ ) {
        int data_id = thread_id + i* total_number_of_threads + offset;
        if( data_id < nz ) {
            float data = values[data_id];
            int row = rIndices[data_id];
            int column = cIndices[data_id];
            float multiplication_value = data * vector[column];
            vals[threadIdx.x] = multiplication_value;
            rows[threadIdx.x] = row;
            __syncthreads();
            int lane = threadIdx.x % 32;
            float val = segmentedScan_16_WarpSize(lane,rows,vals);
            __syncthreads();
            if( lane % 16 == 15) {
                atomicAdd(&result[rows[threadIdx.x]],vals[threadIdx.x]);
            }
            __syncthreads();
        }
    }
}

__global__ void workLoad8Size(int nz, int offset,  int *rIndices, int *cIndices, float *values, int M, int N, float *vector, float *result ) {
    
    int thread_id = threadIdx.x + (blockIdx.x *blockDim.x);
    int total_number_of_threads = blockDim.x * gridDim.x;
    int iteration = nz % total_number_of_threads ? nz/total_number_of_threads + 1 : nz/total_number_of_threads;
    
    __shared__ int rows[1024];
    __shared__ float vals[1024];
    
    int warpId = threadIdx.x >> 5;
    int warpFirst = warpId << 5;
    int warpLast = warpFirst + 31;
    for( int i = 0 ; i < iteration ; i++ ) {
        int data_id = thread_id + i* total_number_of_threads + offset;
        if( data_id < nz ) {
            float data = values[data_id];
            int row = rIndices[data_id];
            int column = cIndices[data_id];
            float multiplication_value = data * vector[column];
            vals[threadIdx.x] = multiplication_value;
            rows[threadIdx.x] = row;
            __syncthreads();
            int lane = threadIdx.x % 32;
            float val = segmentedScan_8_WarpSize(lane,rows,vals);
            __syncthreads();
            if( lane % 8 == 7 ) {
                atomicAdd(&result[rows[threadIdx.x]],vals[threadIdx.x]);
            }
            __syncthreads();
        }
    }
}

__global__ void workLoad4Size(int nz, int offset,  int *rIndices, int *cIndices, float *values, int M, int N, float *vector, float *result ) {
    
    int thread_id = threadIdx.x + (blockIdx.x *blockDim.x);
    int total_number_of_threads = blockDim.x * gridDim.x;
    int iteration = nz % total_number_of_threads ? nz/total_number_of_threads + 1 : nz/total_number_of_threads;
    
    __shared__ int rows[1024];
    __shared__ float vals[1024];
    
    int warpId = threadIdx.x >> 5;
    int warpFirst = warpId << 5;
    int warpLast = warpFirst + 31;
    for( int i = 0 ; i < iteration ; i++ ) {
        int data_id = thread_id + i* total_number_of_threads + offset;
        if( data_id < nz ) {
            float data = values[data_id];
            int row = rIndices[data_id];
            int column = cIndices[data_id];
            float multiplication_value = data * vector[column];
            vals[threadIdx.x] = multiplication_value;
            rows[threadIdx.x] = row;
            __syncthreads();
            int lane = threadIdx.x % 32;
            float val = segmentedScan_4_WarpSize(lane,rows,vals);
            __syncthreads();
            if( lane % 4 == 3 ) {
                atomicAdd(&result[rows[threadIdx.x]],vals[threadIdx.x]);
            }
            __syncthreads();
        }
    }
}

__global__ void workLoad2Size(int nz, int offset,  int *rIndices, int *cIndices, float *values, int M, int N, float *vector, float *result ) {
    
    int thread_id = threadIdx.x + (blockIdx.x *blockDim.x);
    int total_number_of_threads = blockDim.x * gridDim.x;
    int iteration = nz % total_number_of_threads ? nz/total_number_of_threads + 1 : nz/total_number_of_threads;
    
    __shared__ int rows[1024];
    __shared__ float vals[1024];
    
    int warpId = threadIdx.x >> 5;
    int warpFirst = warpId << 5;
    int warpLast = warpFirst + 31;
    for( int i = 0 ; i < iteration ; i++ ) {
        int data_id = thread_id + i* total_number_of_threads + offset;
        if( data_id < nz ) {
            float data = values[data_id];
            int row = rIndices[data_id];
            int column = cIndices[data_id];
            float multiplication_value = data * vector[column];
            vals[threadIdx.x] = multiplication_value;
            rows[threadIdx.x] = row;
            __syncthreads();
            int lane = threadIdx.x % 32;
            float val = segmentedScan_2_WarpSize(lane,rows,vals);
            __syncthreads();
            if( lane % 2 == 1 ) {
                atomicAdd(&result[rows[threadIdx.x]],vals[threadIdx.x]);
            }
            __syncthreads();
        }
    }
}

__global__ void workLoad1Size(int nz, int offset,  int *rIndices, int *cIndices, float *values, int M, int N, float *vector, float *result ) {
    
    int thread_id = threadIdx.x + (blockIdx.x *blockDim.x);
    int total_number_of_threads = blockDim.x * gridDim.x;
    int iteration = nz % total_number_of_threads ? nz/total_number_of_threads + 1 : nz/total_number_of_threads;
    
    int warpId = threadIdx.x >> 5;
    int warpFirst = warpId << 5;
    int warpLast = warpFirst + 31;
    for( int i = 0 ; i < iteration ; i++ ) {
        int data_id = thread_id + i* total_number_of_threads + offset;
        if( data_id < nz ) {
            float data = values[data_id];
            int row = rIndices[data_id];
            int column = cIndices[data_id];
            float multiplication_value = data * vector[column];
            atomicAdd(&result[row],multiplication_value);
        }
    }
}


__global__ void launchKernels(int nz, int * nonZeros, int *rIndices, int *cIndices, float *values, int M, int N, float *vector, float *result, int blockSize, int blockNum){

    workLoad32Size<<blockSize,blockNum>>(nz,nonZeros[0],rIndices,cIndices,values,M,N,vector,result);
    workLoad16Size<<blockSize,blockNum>>(nz,nonZeros[1],rIndices,cIndices,values,M,N,vector,result);
    workLoad8Size<<blockSize,blockNum>>(nz,nonZeros[2],rIndices,cIndices,values,M,N,vector,result);
    workLoad4Size<<blockSize,blockNum>>(nz,nonZeros[3],rIndices,cIndices,values,M,N,vector,result);
    workLoad2Size<<blockSize,blockNum>>(nz,nonZeros[4],rIndices,cIndices,values,M,N,vector,result);
    workLoad1Size<<blockSize,blockNum>>(nz,nonZeros[5],rIndices,cIndices,values,M,N,vector,result);
}


typedef struct rowInfo{
    int rowId;
    int start;
    int end;
    int nz;
}rowInfo;

void getMulDesign(MatrixInfo * mat, MatrixInfo * vec, MatrixInfo * res, int blockSize, int blockNum){
    /*Allocate*/
    
    /* Get the values required from the MatrixInfo Structure */
    int number_of_non_zeros = mat->nz;
    int *row_indices = mat->rIndex;
    int *column_indices = mat->cIndex;
    float *values = mat->val;
    int M = mat->M;
    int N = mat->N;
    float *vector = vec->val;
    float *result = res->val;
    int *nonZeros = (int*)malloc(sizeof(int)*6);
    
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

    rowInfo *rowInformation = (rowInfo*)malloc(sizeof(rowInfo)*M);
    for( int i = 0 ; i < M; i++ ) {
        rowInformation[i]->rowId = i;
        rowInformation[i]->start = -1;
        rowInformation[i]->end = -1;
        rowInformation[i]->nz = 0;
    }
    
    for( int i = 0 ; i < number_of_non_zeros ; i++ ) {
        
        int row = row_indices[i];
        
        if( rowInformation[row]->nz == 0 ) {
            rowInformation[row]->start = i;
        }
        
        if( i < number_of_non_zeros - 1) {
            if( row != row_indices[i+1])
                rowInformation[row]->end = i;
        }
        
        if( i == number_of_non_zeros - 1) {
            rowInformation[row]->end = i;
        }
        
        rowInformation[row]->nz++;
        
    }
    
    int currentPosition = 0;
    int nz_index = 0;
    for( int offset = 32 ; offset > 0 ; offset >>= 1) {
        nonZeros[nz_index] = 0;
        for( int i = 0 ; i < M ; i++ ) {
            int currentNonZeros = rowInformation[i]->nz;
            int number_of_sets;
            
            if( currentNonZeros / offset > 0 ) {
                
                number_of_sets = currentNonZeros/offset;
                int start = rowInformation[row]->start;
                int noElements = number_of_sets*offset;
                for( int j = 0 ; j < noElements ; j++ ) {
                    sorting[currentPosition].row = row_indices[start+j];
                    sorting[currentPosition].column = column_indices[start+j];
                    sorting[currentPosition].value = values[start+j];
                    currentPosition++;
                }
                rowInformation[i]->start = start + noElements;
                rowInformation[i]->nz = rowInformation[i]->nz % 32;
                nonZeros[nz_index] += noElements;
            }
        }
        nz_index++;
    }
    
    for( int i = 0; i < number_of_non_zeros ; i++ ) {
        row_indices[i] = sorting[i].row;
        column_indices[i] = sorting[i].column;
        values[i] = sorting[i].value;
    }
    
    
    printf("\nGPU Code");
    printf("\nBlock Size : %lu, Number of Blocks : %lu, nz : %lu\n",blockSize,blockNum,number_of_non_zeros);
    
    /* Device copies of the required values */
    int   * d_nonZeros
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
    cudaMalloc((void**)&d_nonZeros, sizeof(int)*6);

    /* Set all the result values to be zeros */
    cudaMemset(d_result, 0, sizeof(float)*M);
    
    /* Copying values from host to device */
    cudaMemcpy(d_rIndices,row_indices,sizeof(int)*number_of_non_zeros, cudaMemcpyHostToDevice);
    cudaMemcpy(d_cIndices,column_indices, sizeof(int)*number_of_non_zeros , cudaMemcpyHostToDevice);
    cudaMemcpy(d_values,values, sizeof(float)*number_of_non_zeros,cudaMemcpyHostToDevice);
    cudaMemcpy(d_vector,vector, sizeof(float)*N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_nonZeros,nonZeros,sizeof(int)*6,cudaMemcpyHostToDevice);
    

    
    
    struct timespec start, end;
    cudaDeviceSynchronize();
    clock_gettime(CLOCK_MONOTONIC_RAW, &start);
    /*Your own magic here!*/

    launchKernels<<blockSize,blockNum>>>(number_of_non_zeros,d_nonZeros,d_rIndices,d_cIndices,d_values,M,N,d_vector,d_result,blockSize,blockNum);
    
    cudaDeviceSynchronize();
    clock_gettime(CLOCK_MONOTONIC_RAW, &end);
    printf("Your Own Kernel Time: %lu micro-seconds\n", 1000000 * (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1000);

    /*Deallocate*/
}
