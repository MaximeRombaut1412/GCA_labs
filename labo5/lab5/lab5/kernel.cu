
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cstdlib>
#include <stdio.h>
#include <iostream>
#include <cmath>
# include <chrono>

using namespace std;
#define SIZE 2048
#define THREADS 1024
#define N 1

__global__ void max_reduce(int* inputData, int dataSize, int offset) {
    int idx = threadIdx.x;

    if (idx < dataSize) {
        for (int s = 1; s < dataSize; s *= 2) {
            if (idx < dataSize / (2 * s)) {

                int l = inputData[(idx * 2) + offset];
                int r = inputData[(idx * 2) + 1 + offset];
                inputData[idx + offset] = max(l, r);

            }
            __syncthreads();
        }
    }
    __syncthreads();
    
}
__global__ void min_reduce(int* inputData, int dataSize, int offset) {
    int idx = threadIdx.x;

    if (idx < dataSize) {
        for (int s = 1; s < dataSize; s *= 2) {
            if (idx < dataSize / (2 * s)) {

                int l = inputData[(idx * 2) + offset];
                int r = inputData[(idx * 2) + 1 + offset];
                inputData[idx + offset] = min(l, r);

            }
            __syncthreads();
        }
    }
    __syncthreads();

}

__global__ void sum_reduce(int* arr, int dataSize, int offset) {
    int idx = threadIdx.x;
    if (idx < dataSize) {
        for (int s = 1; s < dataSize; s *= 2) {
            if (idx < dataSize / (2 * s)) {
                int l = arr[(idx * 2) + offset];
                int r = arr[(idx * 2) + 1 + offset];
                arr[idx + offset] = l + r;
            }
            __syncthreads();
        }
    }
    __syncthreads();
}
__global__ void product_reduce(int* arr, int dataSize, int offset) {
    int idx = threadIdx.x;
    if (idx < dataSize) {
        for (int s = 1; s < dataSize; s *= 2) {
            if (idx < dataSize / (2 * s)) {
                int l = arr[(idx * 2) + offset];
                int r = arr[(idx * 2) + 1 + offset];
                arr[idx + offset] = l * r;
            }
            __syncthreads();
        }
    }
    __syncthreads();
}

__global__ void sum_kernel(int* arr, int* result) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < SIZE) {
        atomicAdd(result, arr[tid]);
    }
}

__global__ void product_kernel(int* arr, int* result) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < SIZE) {
        *result *= arr[tid];
    }
}


void sync() {

    /*int* test = (int*)malloc(SIZE* 4 * sizeof(int));
    for (int j = 0; j < 4* N; j++) {
        test[j] = N - j;
    }*/
    const int threads_per_block = THREADS;
    int blocks_per_grid = (SIZE + threads_per_block - 1) / threads_per_block;
    //Sum
    //---------------------------------------------------------------------
    //const auto startSync = std::chrono::steady_clock::now();
    int* arr_sum = (int*)malloc(SIZE * sizeof(int));
    int* arr_sum_res = (int*)malloc(SIZE * sizeof(int));
    for (int i = 0; i < SIZE; ++i) {
        arr_sum[i] = 1;
    }
   
    int* d_arr_sum = NULL;
    int* d_arr_sum_res = NULL;
    cudaMalloc((void**)&d_arr_sum, SIZE * sizeof(int));
    cudaMalloc((void**)&d_arr_sum_res, SIZE * sizeof(int));

    //Product
    //---------------------------------------------------------------------
    int* arr_product = (int*)malloc(SIZE * sizeof(int));
    int* arr_product_res = (int*)malloc(SIZE * sizeof(int));
    for (int i = 0; i < SIZE; ++i) {
        arr_product[i] = 1;
    }

    int* d_arr_product = NULL;
    int* d_arr_product_res = NULL;
    cudaMalloc((void**)&d_arr_product, SIZE * sizeof(int));
    cudaMalloc((void**)&arr_product_res, SIZE * sizeof(int));

    
    //Min
    //---------------------------------------------------------------------
    int* arr_min = (int*)malloc(SIZE * sizeof(int));
    for (int i = 0; i < SIZE; ++i) {
        arr_min[i] = 1;
    }

    int* d_arr_min = NULL;
    cudaMalloc((void**)&d_arr_min, SIZE * sizeof(int));

    //Max
    //---------------------------------------------------------------------
    int* arr_max = (int*)malloc(SIZE * sizeof(int));
    for (int i = 0; i < SIZE; ++i) {
        arr_max[i] = 1;
    }

    int* d_arr_max = NULL;
    cudaMalloc((void**)&d_arr_max, SIZE * sizeof(int));

    /*const auto endSync = std::chrono::steady_clock::now();
    const std::chrono::duration<double> elapsed_seconds{ endSync - startSync };
    cout << "ArrayTime: " << elapsed_seconds.count() << "\n";*/

    //Sum GPU
    //---------------------------------------------------------------------
    cudaMemcpy(d_arr_sum, arr_sum, SIZE * sizeof(int), cudaMemcpyHostToDevice);
    sum_kernel <<<blocks_per_grid, threads_per_block >>> (d_arr_sum, d_arr_sum_res);
    cudaMemcpy(arr_sum_res, d_arr_sum_res, SIZE * sizeof(int), cudaMemcpyDeviceToHost);

    //Product GPU
    //---------------------------------------------------------------------
    cudaMemcpy(d_arr_product, arr_product, SIZE * sizeof(int), cudaMemcpyHostToDevice);
    product_kernel << <blocks_per_grid, threads_per_block >> > (d_arr_product, d_arr_product_res);
    cudaMemcpy(arr_product_res, d_arr_product, SIZE * sizeof(int), cudaMemcpyDeviceToHost);


    //Min GPU
    //---------------------------------------------------------------------
    cudaMemcpy(d_arr_min, arr_min, SIZE * sizeof(int), cudaMemcpyHostToDevice);
    min_reduce << <1, 1024 >> > (d_arr_min, 1024, 0);
    cudaMemcpy(arr_min, d_arr_min, SIZE * sizeof(int), cudaMemcpyDeviceToHost);


    //Max GPU
    //---------------------------------------------------------------------
    cudaMemcpy(d_arr_max, arr_max, SIZE * sizeof(int), cudaMemcpyHostToDevice);

    
    max_reduce << <1, 1024 >> > (d_arr_max, 1024, 0);
    cudaMemcpy(arr_max, d_arr_max, SIZE * sizeof(int), cudaMemcpyDeviceToHost);
    //
    // cout << "test2 \n";

    
    //cudaDeviceSynchronize();
    
    cudaFree(d_arr_sum);
    cudaFree(d_arr_product);
    cudaFree(d_arr_min);
    cudaFree(d_arr_max);
    free(arr_sum);
    //free(test);
    free(arr_product);
    free(arr_min);
    free(arr_max);
}
void sync_async_copy() {

    /*int* test = (int*)malloc(SIZE *4* sizeof(int));
    for (int j = 0; j < 4*N; j++) {
        test[j] = N - j;
    }*/

    const int threads_per_block = THREADS;
    int blocks_per_grid = (SIZE + threads_per_block - 1) / threads_per_block;
    //Sum
    //---------------------------------------------------------------------
    int* arr_sum = (int*)malloc(SIZE * sizeof(int));
    int* arr_sum_res = (int*)malloc(SIZE * sizeof(int));
    for (int i = 0; i < SIZE; ++i) {
        arr_sum[i] = 1;
    }

    int* d_arr_sum = NULL;
    int* d_arr_sum_res = NULL;
    cudaMalloc((void**)&d_arr_sum, SIZE * sizeof(int));
    cudaMalloc((void**)&d_arr_sum_res, SIZE * sizeof(int));

    //Product
    //---------------------------------------------------------------------
    int* arr_product = (int*)malloc(SIZE * sizeof(int));
    int* arr_product_res = (int*)malloc(SIZE * sizeof(int));
    for (int i = 0; i < SIZE; ++i) {
        arr_product[i] = 1;
    }

    int* d_arr_product = NULL;
    int* d_arr_product_res = NULL;
    cudaMalloc((void**)&d_arr_product, SIZE * sizeof(int));
    cudaMalloc((void**)&arr_product_res, SIZE * sizeof(int));


    //Min
    //---------------------------------------------------------------------
    int* arr_min = (int*)malloc(SIZE * sizeof(int));
    for (int i = 0; i < SIZE; ++i) {
        arr_min[i] = 1;
    }

    int* d_arr_min = NULL;
    cudaMalloc((void**)&d_arr_min, SIZE * sizeof(int));

    //Max
    //---------------------------------------------------------------------
    int* arr_max = (int*)malloc(SIZE * sizeof(int));
    for (int i = 0; i < SIZE; ++i) {
        arr_max[i] = 1;
    }

    int* d_arr_max = NULL;
    cudaMalloc((void**)&d_arr_max, SIZE * sizeof(int));

    //Sum GPU
    //---------------------------------------------------------------------
   /* cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);*/

    cudaMemcpyAsync(d_arr_sum, arr_sum, SIZE * sizeof(int), cudaMemcpyHostToDevice);
    sum_kernel << <blocks_per_grid, threads_per_block >> > (d_arr_sum, d_arr_sum_res);
    cudaMemcpyAsync(arr_sum_res, d_arr_sum_res, SIZE * sizeof(int), cudaMemcpyDeviceToHost);
    /*cudaEventSynchronize(start);
    cudaEventRecord(stop);
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    cout << "Cudatime: " << ms << "\n";*/
    cudaDeviceSynchronize();
    //Product GPU
    //---------------------------------------------------------------------
    cudaMemcpyAsync(d_arr_product, arr_product, SIZE * sizeof(int), cudaMemcpyHostToDevice);
    product_kernel << <blocks_per_grid, threads_per_block >> > (d_arr_product, d_arr_product_res);
    cudaMemcpyAsync(arr_product_res, d_arr_product, SIZE * sizeof(int), cudaMemcpyDeviceToHost);


    //Min GPU
    //---------------------------------------------------------------------
    cudaMemcpyAsync(d_arr_min, arr_min, SIZE * sizeof(int), cudaMemcpyHostToDevice);

    /* int offset = 0;

     for (int i = 0; i < ceil(fmax((SIZE / 2048), 1)); i++) {
         if (SIZE - 2048 * i < 2048) {
             min_reduce << <1, SIZE / 2 - 2048 * i >> > (d_arr_min, SIZE - 2048 * i, offset);
         }
         else {
             min_reduce << <1, 1024 >> > (d_arr_min, 2048, offset);
         }
         offset += 2048;
     }*/
    min_reduce << <1, THREADS >> > (d_arr_min, SIZE, 0);
    cudaMemcpyAsync(arr_min, d_arr_min, SIZE * sizeof(int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    //Max GPU
    //---------------------------------------------------------------------
    cudaMemcpy(d_arr_max, arr_max, SIZE * sizeof(int), cudaMemcpyHostToDevice);

    /*offset = 0;

    for (int i = 0; i < ceil(fmax((SIZE / 2048), 1)); i++) {
        if (SIZE - 2048 * i < 2048) {
            max_reduce << <1, SIZE / 2 - 2048 * i >> > (d_arr_max, SIZE - 2048 * i, offset);
        }
        else {
            max_reduce << <1, 1024 >> > (d_arr_max, 2048, offset);
        }
        offset += 2048;
    }*/
    max_reduce << <1, THREADS >> > (d_arr_max, SIZE, 0);
    cudaMemcpy(arr_max, d_arr_max, SIZE * sizeof(int), cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();

    cudaFree(d_arr_sum);
    cudaFree(d_arr_product);
    cudaFree(d_arr_min);
    cudaFree(d_arr_max);
    free(arr_sum);
    //free(test);
    free(arr_product);
    free(arr_min);
    free(arr_max);
}

void async_with_streams() {
    cudaStream_t stream_sum, stream_product, stream_min, stream_max;
    cudaStreamCreate(&stream_sum);
    cudaStreamCreate(&stream_product);
    cudaStreamCreate(&stream_min);
    cudaStreamCreate(&stream_max);
    const int threads_per_block = THREADS;
    int blocks_per_grid = (SIZE + threads_per_block - 1) / threads_per_block;
    //Sum
    //---------------------------------------------------------------------
    int* arr_sum = (int*)malloc(SIZE * sizeof(int));
    int* arr_sum_res = (int*)malloc(SIZE * sizeof(int));
    for (int i = 0; i < SIZE; ++i) {
        arr_sum[i] = 1;
    }
    /*int* test = (int*)malloc(SIZE * sizeof(int));
    for (int j = 0; j < N; j++) {
        test[j] = N - j;
    }*/
    
    int* d_arr_sum = NULL;
    int* d_arr_sum_res = NULL;
    cudaMalloc((void**)&d_arr_sum, SIZE * sizeof(int));
    cudaMalloc((void**)&d_arr_sum_res, SIZE * sizeof(int));

    cudaMemcpyAsync(d_arr_sum, arr_sum, SIZE * sizeof(int), cudaMemcpyHostToDevice, stream_sum);
    sum_kernel << <blocks_per_grid, threads_per_block >> > (d_arr_sum, d_arr_sum_res);
    cudaMemcpyAsync(arr_sum_res, d_arr_sum_res, SIZE * sizeof(int), cudaMemcpyDeviceToHost,stream_sum);
    
    //Product
    //---------------------------------------------------------------------
    int* arr_product = (int*)malloc(SIZE * sizeof(int));
    int* arr_product_res = (int*)malloc(SIZE * sizeof(int));
    for (int i = 0; i < SIZE; ++i) {
        arr_product[i] = 1;
    }
    
    /*for (int j = 0; j < N; j++) {
        test[j] = N - j;
    }*/
    int* d_arr_product = NULL;
    int* d_arr_product_res = NULL;
    
    cudaMalloc((void**)&d_arr_product, SIZE * sizeof(int));
    cudaMalloc((void**)&d_arr_product_res, SIZE * sizeof(int));
    cudaMemcpyAsync(d_arr_product, arr_product, SIZE * sizeof(int), cudaMemcpyHostToDevice, stream_product);
    
    product_kernel << <blocks_per_grid, threads_per_block >> > (d_arr_product, d_arr_product_res);
    cudaMemcpyAsync(arr_product_res, d_arr_product_res, SIZE * sizeof(int), cudaMemcpyDeviceToHost, stream_product);
    
    //Min
    //---------------------------------------------------------------------
    int* arr_min = (int*)malloc(SIZE * sizeof(int));
    for (int i = 0; i < SIZE; ++i) {
        arr_min[i] = 100;
    }
  
    /*for (int j = 0; j < N; j++) {
        test[j] = N - j;
    }*/
    int* d_arr_min = NULL;

    cudaMalloc((void**)&d_arr_min, SIZE * sizeof(int));
    cudaMemcpyAsync(d_arr_min, arr_min, SIZE * sizeof(int), cudaMemcpyHostToDevice, stream_min);

    min_reduce << <1, THREADS >> > (d_arr_min, SIZE, 0);
    cudaMemcpyAsync(arr_min, d_arr_min, SIZE * sizeof(int), cudaMemcpyDeviceToHost,stream_min);

    //Max
    //---------------------------------------------------------------------
    int* arr_max = (int*)malloc(SIZE * sizeof(int));
    for (int i = 0; i < SIZE; ++i) {
        arr_max[i] = 1;
    }
   
    /*for (int j = 0; j < N; j++) {
        test[j] = N - j;
    }*/
    int* d_arr_max = NULL;
   
    cudaMalloc((void**)&d_arr_max, SIZE * sizeof(int));
    cudaMemcpyAsync(d_arr_max, arr_max, SIZE * sizeof(int), cudaMemcpyHostToDevice,stream_max);

    
    max_reduce << <1, THREADS >> > (d_arr_max, SIZE, 0);
    cudaMemcpyAsync(arr_max, d_arr_max, SIZE * sizeof(int), cudaMemcpyDeviceToHost,stream_max);


    cudaStreamSynchronize(stream_sum);
    cudaStreamSynchronize(stream_product);
    cudaStreamSynchronize(stream_min);
    cudaStreamSynchronize(stream_max);
    cudaStreamDestroy(stream_sum);
    cudaStreamDestroy(stream_product);
    cudaStreamDestroy(stream_min);
    cudaStreamDestroy(stream_max);

    cudaDeviceSynchronize();

    cudaFree(d_arr_sum);
    cudaFree(d_arr_product);
    cudaFree(d_arr_min);
    cudaFree(d_arr_max);
    free(arr_sum);
    //free(test);
    free(arr_product);
    free(arr_min);
    free(arr_max);
}

void async_without_streams() {
    const int threads_per_block = THREADS;
    int blocks_per_grid = (SIZE + threads_per_block - 1) / threads_per_block;
    //Sum
    //---------------------------------------------------------------------
    int* arr_sum = (int*)malloc(SIZE * sizeof(int));
    int* arr_sum_res = (int*)malloc(SIZE * sizeof(int));
    for (int i = 0; i < SIZE; ++i) {
        arr_sum[i] = 1;
    }
    /*int* test = (int*)malloc(SIZE * sizeof(int));
    for (int j = 0; j < N; j++) {
        test[j] = N - j;
    }*/
    int* d_arr_sum = NULL;
    int* d_arr_sum_res = NULL;
    cudaMalloc((void**)&d_arr_sum, SIZE * sizeof(int));
    cudaMalloc((void**)&d_arr_sum_res, SIZE * sizeof(int));

    cudaMemcpyAsync(d_arr_sum, arr_sum, SIZE * sizeof(int), cudaMemcpyHostToDevice);
    sum_kernel << <blocks_per_grid, threads_per_block >> > (d_arr_sum, d_arr_sum_res);
    cudaMemcpyAsync(arr_sum_res, d_arr_sum_res, SIZE * sizeof(int), cudaMemcpyDeviceToHost);

    //Product
    //---------------------------------------------------------------------
    int* arr_product = (int*)malloc(SIZE * sizeof(int));
    int* arr_product_res = (int*)malloc(SIZE * sizeof(int));
    for (int i = 0; i < SIZE; ++i) {
        arr_product[i] = 1;
    }
    int* d_arr_product = NULL;
    int* d_arr_product_res = NULL;
   
    /*for (int j = 0; j < N; j++) {
        test[j] = N - j;
    }*/
    cudaMalloc((void**)&d_arr_product, SIZE * sizeof(int));
    cudaMalloc((void**)&d_arr_product_res, SIZE * sizeof(int));
    cudaMemcpyAsync(d_arr_product, arr_product, SIZE * sizeof(int), cudaMemcpyHostToDevice);

    product_kernel << <blocks_per_grid, threads_per_block >> > (d_arr_product, d_arr_product_res);
    cudaMemcpyAsync(arr_product_res, d_arr_product_res, SIZE * sizeof(int), cudaMemcpyDeviceToHost);

    //Min
    //---------------------------------------------------------------------
    int* arr_min = (int*)malloc(SIZE * sizeof(int));
    for (int i = 0; i < SIZE; ++i) {
        arr_min[i] = 100;
    }

    int* d_arr_min = NULL;
    /*for (int j = 0; j < N; j++) {
        test[j] = N - j;
    }*/
    cudaMalloc((void**)&d_arr_min, SIZE * sizeof(int));
    cudaMemcpyAsync(d_arr_min, arr_min, SIZE * sizeof(int), cudaMemcpyHostToDevice);

   /* int offset = 0;

    for (int i = 0; i < ceil(fmax((SIZE / 2048), 1)); i++) {
        if (SIZE - 2048 * i < 2048) {
            min_reduce << <1, SIZE / 2 - 2048 * i >> > (d_arr_min, SIZE - 2048 * i, offset);
        }
        else {
            min_reduce << <1, 1024 >> > (d_arr_min, 2048, offset);
        }
        offset += 2048;
    }*/
    min_reduce << <1, THREADS >> > (d_arr_min, SIZE, 0);
    cudaMemcpyAsync(arr_min, d_arr_min, SIZE * sizeof(int), cudaMemcpyDeviceToHost);

    //Max
    //---------------------------------------------------------------------
    int* arr_max = (int*)malloc(SIZE * sizeof(int));
    for (int i = 0; i < SIZE; ++i) {
        arr_max[i] = 1;
    }

    int* d_arr_max = NULL;
    /*for (int j = 0; j < N; j++) {
        test[j] = N - j;
    }*/
    cudaMalloc((void**)&d_arr_max, SIZE * sizeof(int));
    cudaMemcpyAsync(d_arr_max, arr_max, SIZE * sizeof(int), cudaMemcpyHostToDevice);

    /*offset = 0;

    for (int i = 0; i < ceil(fmax((SIZE / 2048), 1)); i++) {
        if (SIZE - 2048 * i < 2048) {
            max_reduce << <1, SIZE / 2 - 2048 * i >> > (d_arr_max, SIZE - 2048 * i, offset);
        }
        else {
            max_reduce << <1, 1024 >> > (d_arr_max, 2048, offset);
        }
        offset += 2048;
    }*/
    max_reduce << <1, THREADS >> > (d_arr_max, SIZE, 0);
    cudaMemcpyAsync(arr_max, d_arr_max, SIZE * sizeof(int), cudaMemcpyDeviceToHost);


    cudaDeviceSynchronize();
    //free(test);
    cudaFree(d_arr_sum);
    cudaFree(d_arr_product);
    cudaFree(d_arr_min);
    cudaFree(d_arr_max);
    free(arr_sum);

  
    free(arr_product);
    free(arr_min);
    free(arr_max);
}

float kernel_timings() {
    const int threads_per_block = THREADS;
    int blocks_per_grid = (SIZE + threads_per_block - 1) / threads_per_block;
    //Sum
    //---------------------------------------------------------------------


    const auto startSync = std::chrono::steady_clock::now();
  
    int* arr_sum = (int*)malloc(SIZE * sizeof(int));
    int* arr_sum_res = (int*)malloc(SIZE * sizeof(int));
    for (int i = 0; i < SIZE; ++i) {
        arr_sum[i] = 1;
    }
    const auto endSync = std::chrono::steady_clock::now();
    const std::chrono::duration<double> elapsed_seconds{ endSync - startSync };
    float ms2 = elapsed_seconds.count();
    cout << "Execution time sync: " << ms2 << "\n";

    int* d_arr_sum = NULL;
    int* d_arr_sum_res = NULL;
    cudaMalloc((void**)&d_arr_sum, SIZE * sizeof(int));
    cudaMalloc((void**)&d_arr_sum_res, SIZE * sizeof(int));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    

    cudaMemcpyAsync(d_arr_sum, arr_sum, SIZE * sizeof(int), cudaMemcpyHostToDevice);
    cudaEventRecord(start);
    //product_kernel << <blocks_per_grid, threads_per_block >> > (d_arr_sum, d_arr_sum_res);
    max_reduce << <1, THREADS >> > (d_arr_sum, SIZE, 0);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    //std::cout << "Kernel execution time: " << ms << " milliseconds" << std::endl;
    std::cout << ms << std::endl;
    cudaMemcpyAsync(arr_sum_res, d_arr_sum_res, SIZE * sizeof(int), cudaMemcpyDeviceToHost);
    free(arr_sum);
    free(arr_sum_res);
    cudaFree(d_arr_sum);
    cudaFree(d_arr_sum_res);
    return ms2;
}



int main()
{
    /*int n = 1000;
    float total = 0;
    for (int i = 0; i < n; i++) {
        total = kernel_timings();
    }
    float avg = total / n;
    cout << "avg: " << avg;*/
    float* s = (float*)malloc(101 * sizeof(float));
    float* s1 = (float*)malloc(101 * sizeof(float));
    float* s2 = (float*)malloc(101 * sizeof(float));
    float* s3 = (float*)malloc(101 * sizeof(float));
    for (int j = 0; j < 101; j++) {


        int n = N;
        double totalSync = 0;
        double totalSyncA = 0;
        double totalASyncS = 0;
        double totalASyncNS = 0;
        for (int i = 0; i < n; i++) {
            //cout << "Iteration " << i << "\n";
            const auto startSync = std::chrono::steady_clock::now();
            sync();
            const auto endSync = std::chrono::steady_clock::now();
            const std::chrono::duration<double> elapsed_seconds{ endSync - startSync };
            //cout << "Execution time sync: " << elapsed_seconds.count() << "\n";
            totalSync += elapsed_seconds.count();

            const auto startSync2 = std::chrono::steady_clock::now();
            sync_async_copy();
            const auto endSync2 = std::chrono::steady_clock::now();
            const std::chrono::duration<double> elapsed_seconds4{ endSync2 - startSync2 };
            //cout << "Execution time sync: " << elapsed_seconds4.count() << "\n";
            totalSyncA += elapsed_seconds4.count();

            const auto startAsync = std::chrono::steady_clock::now();
            async_with_streams();
            const auto endAsync = std::chrono::steady_clock::now();
            const std::chrono::duration<double> elapsed_seconds2{ endAsync - startAsync };
            //cout << "Execution time async with streams: " << elapsed_seconds2.count() << "\n";
            totalASyncS += elapsed_seconds2.count();

            const auto startAsync2 = std::chrono::steady_clock::now();
            async_without_streams();
            const auto endAsync2 = std::chrono::steady_clock::now();
            const std::chrono::duration<double> elapsed_seconds3{ endAsync2 - startAsync2 };
            //cout << "Execution time async without streams: " << elapsed_seconds3.count() << "\n";
            totalASyncNS += elapsed_seconds3.count();
        }

        /*cout << "Sync: " << (totalSync / n) << "\n";
        cout << "SyncA: " << (totalSyncA / n) << "\n";
        cout << "SyncAS: " << (totalASyncS / n) << "\n";
        cout << "SyncANS: " << (totalASyncNS / n) << "\n";
        cout << "\n";*/
        /*cout  << (totalSync / n) << "\n";
        cout << (totalSyncA / n) << "\n";
        cout << (totalASyncS / n) << "\n";
        cout  << (totalASyncNS / n) << "\n";
        cout << "\n";*/
        s[j] = (totalSync / n);
        s1[j] = (totalSyncA / n);
        s2[j] = (totalASyncS / n);
        s3[j] = (totalASyncNS / n);
    }
    for (int i = 0; i < 101; i++) {
        cout << s[i] << "\n";
    }
    cout << "\n";
    for (int i = 0; i < 101; i++) {
        cout << s1[i] << "\n";
    }
    cout << "\n";
    for (int i = 0; i < 101; i++) {
        cout << s2[i] << "\n";
    }
    cout << "\n";
    for (int i = 0; i < 101; i++) {
        cout << s3[i] << "\n";
    }
    return 0;
}