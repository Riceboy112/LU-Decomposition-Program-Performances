#include <stdio.h>
#include <float.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>
#include <cuda.h>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <time.h>
#include <stdio.h>  
#include <cuda.h>

using namespace std;


#define MAX_BLOCK_DIM 1024
#define MAX_2D_BLOCK_DIM 32
#define MAX_CONCURRENT_BLOCKS 56
#define TILE 16
#define HANDLE_ERROR( err )  ( HandleError( err, __FILE__, __LINE__ ) )

typedef struct 
{
    unsigned int index;
    double value;
} local_max;

void HandleError( cudaError_t err, const char *file, int line ) {
    //
    // Handle and report on CUDA errors.
    //
    if ( err != cudaSuccess ) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ), file, line );

        exit( EXIT_FAILURE );
    }
}

void checkCUDAError( const char *msg, bool exitOnError ) {

    cudaError_t err = cudaGetLastError();

    if( cudaSuccess != err) {
        fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err) );
        if (exitOnError) {
            exit(-37);
        }
    }
}

void cleanupCuda( void ) {
	//
  	// Clean up CUDA resources.
  	//

  	//
  	// Explicitly cleans up all runtime-related resources associated with the 
  	// calling host thread.
  	//
    HANDLE_ERROR(
            cudaThreadExit()
    );
}

void getThreadAmount(unsigned int &blocks, unsigned int &threads, const unsigned int &size) 
{
    // gets the number of blocks needed and adds an extra block if necessary when it is split by MAX_BLOCK_DIM
    blocks = (size / MAX_BLOCK_DIM) + ((size % MAX_BLOCK_DIM ) ? 1 : 0 ) ;
    
    // threads is set to MAX_BLOCK_DIM
    threads = MAX_BLOCK_DIM;

    if (size < threads) 
    {
        // threads that will get allocated by the GPU
        blocks = 1; // will use one block 
        if ( size < 2 ) threads = 2;
        else if ( size < 4 ) threads = 4;
        else if ( size < 8 ) threads = 8;
        else if ( size < 16 ) threads = 16;
        else if ( size < 32 ) threads = 32;
        else if ( size < 64 ) threads = 64;
        else if ( size < 128 ) threads = 128;
        else if ( size < 256 ) threads = 256;
        else if ( size < 512 ) threads = 512;
        else if ( size < 1024 ) threads = 1024;
    }
}

// gets the thread amount 
void FindThreadAmount( unsigned int &blocks, unsigned int &threads, const unsigned int &size ) {
    blocks = (size / MAX_2D_BLOCK_DIM) + ((size % MAX_2D_BLOCK_DIM ) ? 1 : 0 ) ;
    threads = MAX_2D_BLOCK_DIM;
    if (size < threads) 
    {
    	threads = size;
    }
}

__device__ double device_abs( double x ) {
    //
    // Calculate x^y on the GPU.
    //
    return fabs( x );
}

// inverts the coefficient of multiplier
__device__ double device_invert_sign(double x) 
{
    // returns the opposite sign of x
    if ( x < 0.0 ) return 1.0;
    else return -1.0;
}




//
// PLACE GPU KERNELS HERE - BEGIN
//

__global__ void ComputeGaussianElimination(double *A, unsigned int *size, unsigned int *switch_row_ptr ) {
    __shared__ unsigned int N;                  // size of the matrix (NxN)
    __shared__ unsigned int switch_row;           // swap the top row with bottom row 
    double coefficient;                         // the coefficient multiplier
    double lower;                            // the element in the lower row
    double upper;                               // the number in the upper row that we will use to multiply by the coefficient

    unsigned int tx = threadIdx.x;
    unsigned int ty = threadIdx.y;
    unsigned int row = blockDim.x * blockIdx.x + tx;
    unsigned int column = blockDim.y * blockIdx.y + ty;

    if (tx == 0 && ty == 0) 
    {
        N = *size;
        switch_row = *switch_row_ptr;
    }

    __syncthreads();

    if ((column > switch_row && column < N) && (row > switch_row && row < N) ) {
        coefficient = A[row*N + switch_row];
        coefficient *= device_invert_sign(A[row * N + column]); // inverts the coefficient multiplier
        lower = A[row * N + column];
        upper = A[switch_row * N + column];
        A[row * N + column] = lower + coefficient * upper;
    }
}


__global__ void FindCoefficients (double *A, unsigned int *size, unsigned int *switch_row_ptr) 
{
    extern __shared__ double data[];           //  shared memory that holds the data 
    __shared__ unsigned int N;                  // size of the matrix (NxN)
    __shared__ unsigned int switch_row;           // swap the top row with bottom row
    __shared__ double denominator;              // the value in denominator
    double coefficient;

    unsigned int tx_id = threadIdx.x;

    if ( tx_id == 0 ) {
        switch_row = *switch_row_ptr;
        N = *size;
        denominator = A[switch_row * N + switch_row]; 
    }

    __syncthreads();

    unsigned int row = blockDim.x * blockIdx.x + tx_id;
    if (row > switch_row && row < N) 
    {
        data[tx_id] = A[row * N + switch_row];
        coefficient = data[tx_id] / denominator;
        A[row * N + switch_row] = coefficient;
    }
}

__global__ void FindMaxColumn(double *A, unsigned int *size, unsigned int *in_column, local_max *get_data) 
{
    extern __shared__ double data[];           // shared memory that holds the data 
    __shared__ unsigned int partition;    // where the partition begins
    __shared__ unsigned int N;                  // size of the matrix (NxN)
    __shared__ unsigned int column;                  // the column that we are finding the max in

    unsigned int tx_id = threadIdx.x;

    if ( tx_id == 0 ) {
        partition = blockDim.x;
        N = *size;
        column = *in_column;
    }

    __syncthreads();

    int row = blockDim.x * blockIdx.x + tx_id;     
    if (row >= column && row < N) 
    {
        data[tx_id] = A[row * N + column];     // store the matrix value in data
        data[tx_id + partition] = row;        // store the matrix index in data
    }
    else 
    {
        data[tx_id] = 0;     // store the matrix value in data
        data[tx_id + partition] = -1;        // store the matrix index in data
    }

    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if ( tx_id < s ) {
            double left_value = device_abs(data[tx_id]);   // gets the absolute value of the left value
            double right_value = device_abs(data[tx_id + s]);   // gets the absolute value of the right value
            if (left_value < right_value) 
            {
                data[tx_id] = data[tx_id + s];
                data[tx_id + partition] = data[tx_id + s + partition];
            }
        }
        __syncthreads();
    }

    if ( tx_id == 0 ) {
        get_data[blockIdx.x].index = data[partition];
        get_data[blockIdx.x].value = data[0];
    }
}

//
// PLACE GPU KERNELS HERE - END
//


int main( int argc, char* argv[] ) {
    // Determine min, max, mean, mode and standard deviation of array
    unsigned int seed;
    struct timeval start, end;
    double runtime;
    bool singular = false;
    bool toPrint = false;

    if( argc < 2 ) {
        printf( "Format: stats_gpu <size of array> <random seed> <print>\n" );
        printf( "Arguments:\n" );
        printf( "  size of array - This is the size of the array to be generated and processed\n" );
        printf( "                  generator that will generate the contents of the array\n" );
        printf( "                  to be processed\n" );
        exit( 1 );
    }
    //
  	// Get the size of the array to process.
  	//
    unsigned int array_size = atoi( argv[1] );
    unsigned int exceed_array = 11;
    unsigned int array_area = array_size * array_size;
    int print = atoi( argv[3] );
    // Get the print variable
    if (print == 1)
    {
        if (array_size < 10)
        {
            toPrint = atoi( argv[3] );
        }
        else
        {
            toPrint = 0;
        }
    }
    else
    {
        toPrint = 0;
    }


    //
  	// Get the seed to be used 
  	//
    seed = atoi( argv[2] );

    //
  	// Make sure that CUDA resources get cleaned up on exit.
  	//
    atexit( cleanupCuda );
    

    // Allocate the array to be populated
    double *array = (double *) malloc(array_area * sizeof(double));
    
    
    //
  	// Output discoveries from the array.
  	//
  	printf("Statistics for array (%d,%d):\n", array_size, seed );

    // initialize the 2d array
    srand(seed);
    for(int i = 0; i < array_size; i++) 
    {
        for ( int j = 0; j < array_size; ++j ) 
        {
            array[i * array_size + j] = ((double) rand() / (double) RAND_MAX);
        }
    }

    if (toPrint == true)
    {
        printf("Matrix A\n");
        for(int i = 0; i < array_size; i++) 
        {
            printf("Row %d: ", i);
            for (int j = 0; j < array_size; ++j) 
            {
                printf("%.2f\t", array[i * array_size + j]);
            }
            printf("\n");
        }
    }
    
    
    runtime = clock()/(float)CLOCKS_PER_SEC;

    // write the matrix  to GPU
    double *A; 
    HANDLE_ERROR(cudaMalloc((void **)&A, array_area * sizeof(double)));
    HANDLE_ERROR(cudaMemcpy(A, array, array_area * sizeof(double), cudaMemcpyHostToDevice));

    // write the array size to GPU
    unsigned int *array_size_ptr;
    HANDLE_ERROR(cudaMalloc((void **)&array_size_ptr, sizeof(unsigned int)));
    HANDLE_ERROR(cudaMemcpy(array_size_ptr, &array_size, sizeof(unsigned int), cudaMemcpyHostToDevice));

    unsigned int *row_ptr;
    HANDLE_ERROR(cudaMalloc((void **)&row_ptr, sizeof(unsigned int)));

    unsigned int *current_row_ptr;
    HANDLE_ERROR(cudaMalloc((void **)&current_row_ptr, sizeof(unsigned int)));


    // -----------------------------------------------------------------------------------------------------------------
    // LU Decomp
    // -----------------------------------------------------------------------------------------------------------------

    unsigned int numblocks, numthreads, data_size;
    getThreadAmount(numblocks, numthreads, array_size);
    local_max *max_threads = new local_max[numblocks];
    local_max *max_threads_ptr;
    
    HANDLE_ERROR(cudaMalloc((void **)&max_threads_ptr, numblocks * sizeof(local_max)));

    for(unsigned int i = 0; i < array_size - 1; ++i) 
    {

        // copy the current column from host to gpu
        HANDLE_ERROR(cudaMemcpy(current_row_ptr, &i, sizeof(unsigned int), cudaMemcpyHostToDevice));

        // find the max for the column so that we can pivot
        getThreadAmount(numblocks, numthreads, array_size);
        data_size = numthreads * 2;
        FindMaxColumn <<< numblocks, numthreads, data_size * sizeof(double) >>>(A, array_size_ptr, current_row_ptr, max_threads_ptr);

        // wait for the gpu to finish
        cudaDeviceSynchronize();

        HANDLE_ERROR(cudaMemcpy(max_threads, max_threads_ptr, numblocks * sizeof(local_max), cudaMemcpyDeviceToHost));

        unsigned int find_index = max_threads[0].index;
        double find_value = max_threads[0].value;
        for (int j = 0; j < numblocks; ++j) 
        {
            if (fabs(max_threads[j].value) > fabs(find_value) && max_threads[j].index != -1.0) 
            {
                find_index = max_threads[j].index;
                find_value = max_threads[j].value;
            }
        }

        if (find_value == 0.0) {
            singular = true;
            break;
        }
        
        if (array_size > exceed_array)
        {
        	//printf("Cannot exceed given array size")
        	break;
        }
        FindCoefficients<<< numblocks, numthreads, numthreads * sizeof(double) >>>(A, array_size_ptr, current_row_ptr);

        // wait for the gpu to finish
        cudaDeviceSynchronize();

        FindThreadAmount(numblocks, numthreads, array_size);

        int numblock = array_size/TILE + ((array_size%TILE)?1:0);
        dim3 grid(numblocks, numblocks);
        //dim3 block(numthreads, numthreads);
        dim3 block(TILE,TILE);

        ComputeGaussianElimination<<< grid, block >>>(A, array_size_ptr, current_row_ptr);

        // wait for the gpu to finish
        cudaDeviceSynchronize();

        HANDLE_ERROR(cudaMemcpy(array, A, array_area * sizeof(double), cudaMemcpyDeviceToHost));

    }
    runtime = clock() - runtime; // calculates the time it took to perform LU decomposition

    // -----------------------------------------------------------------------------------------------------------------
    // End LU Decomp
    // -----------------------------------------------------------------------------------------------------------------

	if (singular == true)
	{
		printf("The matrix is singular!\n");
	}
    if ( singular == false ) {
        HANDLE_ERROR(cudaMemcpy(array, A, array_area * sizeof(double), cudaMemcpyDeviceToHost));

        if (toPrint == true)
        {
            printf("Matrix L:\n");
            for (int i = 0; i < array_size; i++) 
            {
                printf("Row %d: ", i);
                for (int j = 0; j < array_size; ++j) 
                {
                    if ( j < i ) 
                    {
                        printf("%.2f\t", array[i * array_size + j]);
                    } 
                    else if ( j == i ) 
                    {
                        printf("%.2f\t", 1.0);
                    } 
                    else 
                    {
                        printf("%.2f\t", 0.0);
                    }
                }
                printf("\n");
            }

            printf("Matrix U:\n");
            for (int i = 0; i < array_size; i++) {
                printf("Row %d: ", i);
                for (int j = 0; j < array_size; ++j) {
                    if ( j >= i ) {
                        printf("%.2f\t", array[i * array_size + j]);
                    } else {
                        printf("%.2f\t", 0.0);
                    }
                }
                printf("\n");
            }
        }
    }

    // calculate the runtime.
    cout << "Program runs in " << setiosflags(ios::fixed) << setprecision(2) << (runtime)/float(CLOCKS_PER_SEC) << " seconds\n";///////////////////////

    // free cuda memory
    cudaFree(array_size_ptr);
    cudaFree(current_row_ptr);
    cudaFree(A);
    cudaFree(row_ptr);
    cudaFree(max_threads_ptr);

    return 0;
}
