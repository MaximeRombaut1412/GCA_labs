#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include "math.h"
#include <time.h>
using namespace std;

__global__ void vectorFlip(const float* input, float* output, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements)
    {
        output[numElements - i - 1] = input[i];
    }
}

double runFlipGPU(int numElements)
{
    //initialize cpu time dependencies
    clock_t start, end;

    printf("[GPU Vector flip of %d elements]\n", numElements);


    // Print the vector length to be used, and compute its size-
    size_t size = numElements * sizeof(float);

    //printf("allocate %d bytes of memory twice on HOST\n", size);

    // Allocate the host input
    float* h_input = (float*)malloc(size);

    // Allocate the host output
    float* h_output = (float*)malloc(size);

    //printf("fill up A with random integers\n");

    // Initialize the host input vector
    for (int i = 0; i < numElements; ++i) {
        h_input[i] = rand() / (float)RAND_MAX;
    }

    //printf("allocate %d bytes of memory twice on GPU\n", size);

    // Allocate the device input vector
    float* d_input = NULL;
    cudaMalloc((void**)&d_input, size);


    // Allocate the device output vector
    float* d_output = NULL;
    cudaMalloc((void**)&d_output, size);

    // Copy the host input vector in host memory to the device input
    //printf("Copy input data from the host memory to the CUDA device\n");
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);

    // Launch the Vector flip CUDA Kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;

    //start clock measurement
    start = clock();

    //int blocksPerGrid = 196;
    //printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
    vectorFlip << <blocksPerGrid, threadsPerBlock >> > (d_input, d_output, numElements);

    cudaDeviceSynchronize();

    //end clock measurement
    end = clock();

    // Copy the device result vector in device memory to the host result vector
    // in host memory.
    //printf("Copy output data from the CUDA device to the host memory\n");
    cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);

    //print input vector
    //for (int i = 0; i < numElements; i++)
    //{
    //    printf("%.6f ", h_input[i]);
    //}

    //printf("\nflipped vector \n");
    ////print output vector
    //for (int i = 0; i < numElements; i++)
    //{
    //    printf("%.6f ", h_output[i]);
    //}

    //printf("\n");

    // Verify that the result vector is correct
    for (int i = 0; i < numElements; ++i)
    {
        if (h_output[i] != h_input[numElements - i - 1])
        {
            fprintf(stderr, "Result verification failed at element %d!\n", i);
            exit(EXIT_FAILURE);
        }
    }

    printf("Test PASSED\n");

    // Free device global memory
    cudaFree(d_input);
    cudaFree(d_output);

    // Free host memory
    free(h_input);
    free(h_output);

    return ((double)(end - start)) / CLOCKS_PER_SEC;
}

double runFlipCPU(int amount)
{
    //initialize cpu time dependencies
    clock_t start, end;

    printf("[CPU Vector flip of %d elements]\n", amount);

    //allocate memory fore array A and B
    size_t size = amount * sizeof(float);
    float* A = (float*)malloc(size);
    float* B = (float*)malloc(size);

    // Initialize array A and fill it up with random ints
    for (int i = 0; i < amount; ++i) {
        A[i] = rand() / (float)RAND_MAX;
    }

    //start clock measurement
    start = clock();

    //flip array A into array B
    for (int i = 0; i < amount; i++)
    {
        B[i] = A[amount - 1 - i];
    }

    //end clock measurement
    end = clock();

    // Verify that the result vector is correct
    for (int i = 0; i < amount; ++i)
    {
        if (B[i] != A[amount - i - 1])
        {
            fprintf(stderr, "Result verification failed at element %d!\n", i);
            exit(EXIT_FAILURE);
        }
    }

    printf("Test PASSED\n");

    free(A);
    free(B);

    return ((double)(end - start)) / CLOCKS_PER_SEC;
}

int main()
{
    const int base = 10;
    for (int i = 0; i < 9; i++)
    {
        printf("needed clocks: %lf\n", runFlipCPU((int)base * pow(10, i)));
        printf("needed clocks: %lf\n", runFlipGPU((int)base * pow(10, i)));
    }

    //printf("needed clocks: %lf\n", runFlipCPU(5000000000));
    //printf("needed clocks: %lf\n", runFlipGPU(5000000000));

    return 0;
}