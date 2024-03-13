#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cstdint>      // Data types
#include <iostream>     // File operations
#include <chrono>
#include "kernel.h"
using namespace std;

// #define M 512       // Lenna width
// #define N 512       // Lenna height
#define M 941       // VR width
#define N 704       // VR height
#define C 3         // Colors
#define OFFSET 15   // Header length
#define TOTAL M*N*C
#define DIM 2048
#define THREADS 32
#define SHMEM_SIZE 1024 

uint8_t* get_image_array(void) {
    /*
     * Get the data of an (RGB) image as a 1D array.
     *
     * Returns: Flattened image array.
     *
     * Noets:
     *  - Images data is flattened per color, column, row.
     *  - The first 3 data elements are the RGB components
     *  - The first 3*M data elements represent the firts row of the image
     *  - For example, r_{0,0}, g_{0,0}, b_{0,0}, ..., b_{0,M}, r_{1,0}, ..., b_{b,M}, ..., b_{N,M}
     *
     */
     // Try opening the file
    FILE* imageFile;
    imageFile = fopen("D:\\Ku_Leuven\\2023-2024\\Geavanceerde_computerarchitectuur\\labo4\\Lab4\\Lab4\\image-import\\input_image.ppm", "rb");
    if (imageFile == NULL) {
        perror("ERROR: Cannot open output file");
        exit(EXIT_FAILURE);
    }

    // Initialize empty image array
    uint8_t* image_array = (uint8_t*)malloc(M * N * C * sizeof(uint8_t) + OFFSET);

    // Read the image
    fread(image_array, sizeof(uint8_t), M * N * C * sizeof(uint8_t) + OFFSET, imageFile);

    // Close the file
    fclose(imageFile);

    // Move the starting pointer and return the flattened image array
    return image_array + OFFSET;
}

void save_image_grey(uint8_t* image_array)
{
    // Try opening the file
    FILE* imageFile;
    imageFile = fopen("D:\\Ku_Leuven\\2023-2024\\Geavanceerde_computerarchitectuur\\labo4\\Lab4\\Lab4\\image-import\\output_image.ppm", "wb");
    if (imageFile == NULL) {
        perror("ERROR: Cannot open output file");
        exit(EXIT_FAILURE);
    }

    // Configure the file
    fprintf(imageFile, "P5\n");               // P5 filetype
    fprintf(imageFile, "%d %d\n", M, N);      // dimensions
    fprintf(imageFile, "255\n");              // Max pixel

    // Write the image
    fwrite(image_array, 1, M * N, imageFile);

    // Close the file
    fclose(imageFile);
}


void save_image_array(uint8_t* image_array) {
    /*
     * Save the data of an (RGB) image as a pixel map.
     *
     * Parameters:
     *  - param1: The data of an (RGB) image as a 1D array
     *
     */
     // Try opening the file
    FILE* imageFile;
    imageFile = fopen("D:\\Ku_Leuven\\2023-2024\\Geavanceerde_computerarchitectuur\\labo4\\Lab4\\Lab4\\image-import\\output_image.ppm", "wb");
    if (imageFile == NULL) {
        perror("ERROR: Cannot open output file");
        exit(EXIT_FAILURE);
    }

    // Configure the file
    fprintf(imageFile, "P6\n");               // P6 filetype
    fprintf(imageFile, "%d %d\n", M, N);      // dimensions
    fprintf(imageFile, "255\n");              // Max pixel

    // Write the image
    fwrite(image_array, 1, M * N * C, imageFile);
    //fwrite(image_array, 1, M * N, imageFile);
    // Close the file
    fclose(imageFile);
}

//__constant__ int matrices[DIM * DIM * 2];

__global__ void coalesced_access(uint8_t* image_array, uint8_t* output_image) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    output_image[idx] = (image_array[idx*3] + image_array[idx*3 + 1 ] + image_array[idx*3+2]) / 3;
}

__global__ void uncoalesced_access(uint8_t* image_array, uint8_t* output_image) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    output_image[idx] = (image_array[idx] + image_array[idx + N*M] + image_array[idx + 2*N * M]) / 3;
}

__global__ void matrix_multiplication_global(int* a, int* b, int* c) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if ((row < DIM) && (col < DIM)) {
        //c[row * DIM + col] = 0;
        for (int i = 0; i < DIM; i++) {
            c[row * DIM + col] += a[row * DIM + i] * b[i * DIM + col];

        }
    }
}

__global__ void matrix_multiplication_shared_3(int* a, int* b, int* c, int tile_size) {
    __shared__ int A[SHMEM_SIZE];
    __shared__ int B[SHMEM_SIZE];

    int row = blockIdx.y * tile_size + threadIdx.y;
    int col = blockIdx.x * tile_size + threadIdx.x;



    int sum = 0;
    for (int i = 0; i < (DIM / tile_size); i++) {
        A[threadIdx.y * tile_size + threadIdx.x] = a[row * DIM + (i * tile_size + threadIdx.x)];
        B[threadIdx.y * tile_size + threadIdx.x] = b[i * tile_size * DIM + threadIdx.y * DIM + col];
        __syncthreads();

        for (int j = 0; j < tile_size; j++) {
            sum += A[(threadIdx.y * tile_size) + j] * B[j * tile_size + threadIdx.x];
        }
        __syncthreads();
    }
    c[(row * DIM) + col] = sum;

}


//__global__ void matrix_multiplication_constant(int* c) {
//    int idx = blockIdx.x * blockDim.x + threadIdx.x;
//    if (idx < DIM * DIM) {
//        int row = idx / DIM;
//        int col = idx % DIM;
//
//        int sum = 0;
//        for (int k = 0; k < DIM; ++k) {
//            sum += matrices[row * DIM + k] * matrices[(k * DIM + col)+DIM*DIM];
//        }
//        c[row * DIM + col] = sum;
//    }
//}



void uncoalesced_memory_access(int threadsPerBlock) {


    //for (int i = 0; i < 11; i++) {

        uint8_t* image_array = get_image_array();
        uint8_t* reformatted_image = (uint8_t*)malloc(TOTAL);

        for (int i = 0; i < M * N; i++) {
            reformatted_image[i] = image_array[i * 3];
            reformatted_image[(M * N) + i] = image_array[i * 3 + 1];
            reformatted_image[(M * N * 2) + i] = image_array[i * 3 + 2];
        }

        //int threadsPerBlock = 1024;
        int blocksPerGrid = (M * N) / threadsPerBlock;
        int warps = ceil(threadsPerBlock / 32) * blocksPerGrid;

        uint8_t* GPUarray = NULL;
        uint8_t* output_array = NULL;

        cudaMalloc((void**)&output_array, M * N * sizeof(uint8_t));

        cudaMalloc((void**)&GPUarray, TOTAL * sizeof(uint8_t));
        cudaMemcpy(GPUarray, reformatted_image, TOTAL * sizeof(uint8_t), cudaMemcpyHostToDevice);
        
        const auto start = std::chrono::steady_clock::now();
        uncoalesced_access << <blocksPerGrid, threadsPerBlock >> > (GPUarray, output_array);
        const auto end = std::chrono::steady_clock::now();
        const std::chrono::duration<double, milli> elapsed_seconds{ end - start };
        //cout << "Uncoalesced acces " << "Amount of threads: " << threadsPerBlock << ", time: " << elapsed_seconds.count() << "\n";
        cout << elapsed_seconds.count() << "\n";
        cudaFree(GPUarray);

        uint8_t* output_image = (uint8_t*)malloc(M * N);
        cudaMemcpy(output_image, output_array, M * N * sizeof(uint8_t), cudaMemcpyDeviceToHost);

        save_image_grey(output_image);
        cudaFree(output_array);
        free(output_image);
        free(reformatted_image);
    //}
}

void coalesced_memory_access(int threadsPerBlock) {

        int blocksPerGrid = (M * N) / threadsPerBlock;
        uint8_t* image_array = get_image_array();

        uint8_t* GPUarray = NULL;
        uint8_t* output_array = NULL;

        cudaMalloc((void**)&output_array, M * N * sizeof(uint8_t));

        cudaMalloc((void**)&GPUarray, TOTAL * sizeof(uint8_t));
        cudaMemcpy(GPUarray, image_array, TOTAL * sizeof(uint8_t), cudaMemcpyHostToDevice);
        const auto start = std::chrono::steady_clock::now();
        coalesced_access << <blocksPerGrid, threadsPerBlock >> > (GPUarray, output_array);
        const auto end = std::chrono::steady_clock::now();
        cudaFree(GPUarray);

        uint8_t* output_image = (uint8_t*)malloc(M * N);
        cudaMemcpy(output_image, output_array, M * N * sizeof(uint8_t), cudaMemcpyDeviceToHost);

        save_image_grey(output_image);
        cudaFree(output_array);
        free(output_image);


        const std::chrono::duration<double, milli> elapsed_seconds{ end - start };
        //cout << "Coalesced acces " << "Amount of threads: " << threadsPerBlock << ", time: " << elapsed_seconds.count() << "\n";
        cout << elapsed_seconds.count() << "\n";
}


void matrix_multiplication_cpu() {
    const int totalSize = DIM * DIM;
    int* a = (int*)malloc(totalSize * sizeof(int));
    int* b = (int*)malloc(totalSize * sizeof(int));
    //int a[n], b[n];
    for (int i = 1; i < totalSize + 1; i++) {
        a[i - 1] = i;
        b[i - 1] = totalSize - i + 1;
    }
    int arrayLength = (sizeof(a) / sizeof(a[0]));
    int rowLength = DIM;
    int amountOfRows = rowLength;


    const auto start = std::chrono::steady_clock::now();


    int total = 0;
    for (int i = 0; i < rowLength; i++) {
        for (int k = 0; k < rowLength; k++) {
            int value = 0;
            for (int j = 0; j < rowLength; j++) {
                value += a[j + i * rowLength] * b[k + j * rowLength];
            }
            total += value;
        }
    }
    const auto end = std::chrono::steady_clock::now();
    const std::chrono::duration<double, milli> elapsed_seconds{ end - start };
    cout << elapsed_seconds.count() << "\n";
    cout << "Total: " << total << "\n";
}

void shared_multiplication(bool check) {
    int totalSize = DIM * DIM;
    int* a = (int*)malloc(totalSize * sizeof(int));
    int* b = (int*)malloc(totalSize * sizeof(int));
    //int a[n], b[n];
    for (int i = 1; i < totalSize + 1; i++) {
        a[i - 1] = i;
        b[i - 1] = totalSize - i + 1;
    }


    int* c = (int*)malloc(totalSize * sizeof(int));

    int* gpu_a = NULL;
    int* gpu_b = NULL;
    int* gpu_c = NULL;

    const auto start = std::chrono::steady_clock::now();

    cudaMalloc((void**)&gpu_a, totalSize * sizeof(int));
    cudaMalloc((void**)&gpu_b, totalSize * sizeof(int));
    cudaMalloc((void**)&gpu_c, totalSize * sizeof(int));

    cudaMemcpy(gpu_a, a, totalSize * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_b, b, totalSize * sizeof(int), cudaMemcpyHostToDevice);

    int threadsPerBlock = THREADS;
    int blocksPerGrid = DIM/threadsPerBlock;
    dim3 grid(blocksPerGrid, blocksPerGrid);
    dim3 threads(threadsPerBlock, threadsPerBlock);

    //matrix_multiplication_shared_3<<<grid,threads>>> (gpu_a, gpu_b, gpu_c,threadsPerBlock);
    //cout << "Test: " << blocksPerGrid << "\n";


    
    //const auto start = std::chrono::steady_clock::now();
    matrix_multiplication_shared_3 << <grid, threads >> > (gpu_a, gpu_b, gpu_c, threadsPerBlock);
    //const auto end = std::chrono::steady_clock::now();
    cudaMemcpy(c, gpu_c, totalSize * sizeof(int), cudaMemcpyDeviceToHost);
    const auto end = std::chrono::steady_clock::now();
    const std::chrono::duration<double, milli> elapsed_seconds{ end - start };
    if (check) {
        int test = 0;
        for (int i = 0; i < totalSize; i++) {
            //cout << c[i] << "\n";
            test += c[i];
        }
        cout << "Value: " << test << "\n";
    }

    cudaFree(gpu_a);
    cudaFree(gpu_b);
    cudaFree(gpu_c);
    free(c);

    cout << elapsed_seconds.count() << "\n";


}

//void constant_multiplication() {
//    const int n = DIM * DIM;
//    int a[n], b[n];
//    for (int i = 1; i < n + 1; i++) {
//        a[i - 1] = i;
//        b[i - 1] = n - i + 1;
//    }
//
//
//    int totalSize = DIM * DIM;
//    int* c = (int*)malloc(totalSize * sizeof(int));
//
//    int threadsPerBlock = DIM * DIM;
//    if (DIM * DIM > 1024) {
//        threadsPerBlock = 1024;
//    }
//    int blocksPerGrid = ceil(DIM * DIM / threadsPerBlock);
//
//    int combined_matrices[DIM * DIM *2];
//    std::copy(a, a + totalSize, combined_matrices);
//    std::copy(b,b + totalSize, combined_matrices + totalSize);
//
//    int* gpu_c = NULL;
//
//    const auto start = std::chrono::steady_clock::now();
//
//    cudaMalloc((void**)&gpu_c, totalSize * sizeof(int));
//
//    cudaMemcpyToSymbol(matrices, combined_matrices, sizeof(int) * DIM * DIM * 2);
//    matrix_multiplication_constant << <blocksPerGrid, threadsPerBlock >> > (gpu_c);
//    cudaMemcpy(c, gpu_c, totalSize * sizeof(int), cudaMemcpyDeviceToHost);
//
//    const auto end = std::chrono::steady_clock::now();
//    const std::chrono::duration<double, milli> elapsed_seconds{ end - start };
//    //int test = 0;
//    //for (int i = 0; i < totalSize; i++) {
//    //    //cout << c[i] << "\n";
//    //    test += c[i];
//    //}
//    //cout << "Value: " << test << "\n";
//    cudaFree(gpu_c);
//    free(c);
//    cout << elapsed_seconds.count() << "\n";
//}

void global_multiplication(bool check) {
    ;
    const int totalSize = DIM * DIM;
    int* a = (int*)malloc(totalSize * sizeof(int));
    int* b = (int*)malloc(totalSize * sizeof(int));
    //int a[n], b[n];
    for (int i = 1; i < totalSize + 1; i++) {
        a[i - 1] = i;
        b[i - 1] = totalSize - i + 1;
    }

    int threadsPerBlock = 32;
    int blocksPerGrid = DIM / threadsPerBlock;

    dim3 grid(blocksPerGrid, blocksPerGrid);
    dim3 threads(threadsPerBlock, threadsPerBlock);
   
    
    //int totalSize = n;
    int *c = (int*)malloc(totalSize * sizeof(int));
    /*for (int i = 0; i < totalSize; i++) {
        cout << a[i] << "\n";
    }
    for (int i = 0; i < totalSize; i++) {
        cout << b[i] << "\n";
    }*/
    //int threadsPerBlock = 1024 / totalSize;
    //int blocksPerGrid = TOTAL / threadsPerBlock;
    //int warps = ceil(threadsPerBlock / 32) * blocksPerGrid;

    int* gpu_a = NULL;
    int* gpu_b = NULL;
    int* gpu_c = NULL;
    
    const auto start = std::chrono::steady_clock::now();
    cudaMalloc((void**)&gpu_a, totalSize * sizeof(int));
    cudaMalloc((void**)&gpu_b, totalSize * sizeof(int));
    cudaMalloc((void**)&gpu_c, totalSize * sizeof(int));

    cudaMemcpy(gpu_a, a,totalSize * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_b, b, totalSize * sizeof(int), cudaMemcpyHostToDevice);
    //const auto start = std::chrono::steady_clock::now();
   
    matrix_multiplication_global << <grid, threads >> > (gpu_a, gpu_b, gpu_c);
    /*const auto end = std::chrono::steady_clock::now();
    const std::chrono::duration<double, milli> elapsed_seconds{ end - start };*/
    cudaMemcpy(c, gpu_c, totalSize * sizeof(int), cudaMemcpyDeviceToHost);
    const auto end = std::chrono::steady_clock::now();
    const std::chrono::duration<double, milli> elapsed_seconds{ end - start };

    if (check) {
        int test = 0;
        for (int i = 0; i < totalSize; i++) {
            //cout << c[i] << "\n";
            test += c[i];
        }
        cout << "Value: " << test << "\n";
    }
    cudaFree(gpu_a);
    cudaFree(gpu_b);
    cudaFree(gpu_c);
    free(c);
    cout << elapsed_seconds.count() << "\n";
}


int main()
{
    //matrix_multiplication_cpu();
    /*uncoalesced_memory_access(1024);
    for (int i = 0; i < 11; i++) {
        cout << "Uncoalesced , Threads: " << pow(2, i) << "\n";
        for (int j = 0; j < 20; j++) {
            uncoalesced_memory_access(pow(2, i));
        }
    }
    coalesced_memory_access(1024);
    for (int i = 0; i < 11; i++) {
        cout << "Coalesced , Threads: " << pow(2, i) << "\n";
        for (int j = 0; j < 20; j++) {
            coalesced_memory_access(pow(2, i));
        }
    }*/

    //matrix_multiplication_cpu();
    //int a[] = { 1, 2, 3, 4, 5 , 6, 7, 8, 9 };
    //int b[] = { 9,8,7,6,5,4,3,2,1 };
    //int a[] = { 1,2,3,4 };
    //int b[] = { 5,6,7,8 };

    cout << "Matrix size: " << DIM * DIM << "\n";
    //cout << "test123";
    global_multiplication(true);
    cout << "GLOBAL \n";
    for (int i = 0; i < 20; i++) {
        global_multiplication(false);
    }
    cout << "SHARED \n";
    shared_multiplication(true);
    cout << "FIRST \n";
    for (int i = 0; i < 20; i++) {
        shared_multiplication(false);
    }

    
    /*cout << "CONSTANT \n";
    constant_multiplication();
    cout << "FIRST \n";
    for (int i = 0; i < 20; i++) {
        constant_multiplication();
    }*/
    /*global_multiplication();
    shared_multiplication();
    constant_multiplication();*/
    //free(a);
    //free(b);
    return 0;
}