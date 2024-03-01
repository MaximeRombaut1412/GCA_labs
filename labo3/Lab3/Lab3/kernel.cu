/*
 * Code snippet for importing / exporting image data.
 *
 * To convert an image to a pixel map, run `convert <name>.<extension> <name>.ppm
 *
 */
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cstdint>      // Data types
#include <iostream>     // File operations
#include <chrono>
using namespace std;

 // #define M 512       // Lenna width
 // #define N 512       // Lenna height
#define M 941       // VR width
#define N 704       // VR height
#define C 3         // Colors
#define OFFSET 15   // Header length
#define TOTAL M*N*C

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
    imageFile = fopen("D:\\Ku_Leuven\\2023-2024\\Geavanceerde_computerarchitectuur\\labo3\\Lab3\\Lab3\\image-import\\input_image.ppm", "rb");
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



__global__ void inverseImageNoStriding(uint8_t* image_array) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    image_array[idx] = 255 - image_array[idx];

}
__global__ void inverseImageStriding(uint8_t* image_array, int threads) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (TOTAL > threads) {
        int s = (int)ceil(((float)(TOTAL) / threads));
        for (int i = 0; i < s; i++) {
            if (idx +threads * i < TOTAL) {
                image_array[idx + threads * i] = 255 - image_array[idx + threads * i];
            }
        }
    }
    else {
        image_array[idx] = 255 - image_array[idx];
    }
}
__global__ void inverseImageColors(uint8_t* image_array, int threads) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int third = M * N;

    if (TOTAL > threads) {
        int s = (int)ceil(((float)(TOTAL) / threads));
        for (int i = 0; i < s; i++) {
            if (idx + threads * i < TOTAL) {
                if ((idx + threads * i) < third) {
                    /*if ((image_array[idx + threads * i] % 25) * 10 < 150) {
                        image_array[idx + threads * i] = 255;
                    }
                    else {
                        image_array[idx + threads * i] = 255;
                    }*/
                    image_array[idx + threads * i] = (image_array[idx + threads * i] % 25) * 10;
                    //if (image_array[idx + threads * i] > 99 && image_array[idx + threads * i] < 201) { //Moeten we indien kleiner dan 100 en groter dan 200 deze gelijk stellen aan 100/200?
                    //    image_array[idx + threads * i] = (image_array[idx + threads * i] % 25) * 10;
                    //}
                    //else {
                    //    image_array[idx + threads * i] = 255 - image_array[idx + threads * i];
                    //}
                   /* if (image_array[idx + threads * i] > 99 && image_array[idx + threads * i] < 201) {
                        image_array[idx + threads * i] = (image_array[idx + threads * i] % 25) * 10;
                    }
                    else {
                        if (image_array[idx + threads * i] < 100) {
                            image_array[idx + threads * i] = (100 % 25) * 10;
                        }
                        else {
                            image_array[idx + threads * i] = (200 % 25) * 10;
                        }
                    }*/
                }
                else {
                    image_array[idx + threads * i] = 255 - image_array[idx + threads * i];

                }
            }
        }
    }
    else {
        image_array[idx] = 255 - image_array[idx];
    }
}
__global__ void inverseImageRComponent(uint8_t* image_array, int threads) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (TOTAL > threads) {
        int s = (int)ceil(((float)(TOTAL) / threads));
        for (int i = 0; i < s; i++) {
            if (idx + threads * i < TOTAL) {
                if ((idx + threads * i) % 3 == 0) {
                    image_array[idx + threads * i] = (image_array[idx + threads * i] % 25) * 10;

                }
                else {
                    image_array[idx + threads * i] = 255 - image_array[idx + threads * i];

                }
            }
        }
    }


    else {
        image_array[idx] = 255 - image_array[idx];
    }
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
    imageFile = fopen("D:\\Ku_Leuven\\2023-2024\\Geavanceerde_computerarchitectuur\\labo3\\Lab3\\Lab3\\image-import\\output_image.ppm", "wb");
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

    // Close the file
    fclose(imageFile);
}

void noStriding() {
    uint8_t* image_array = get_image_array();
    uint8_t* new_image_array = (uint8_t*)malloc(TOTAL);

    int threadsPerBlock = 1;
    int blocksPerGrid = TOTAL / threadsPerBlock;
    int warps = ceil(threadsPerBlock / 32) * blocksPerGrid;

    uint8_t* GPUarray = NULL;
    cudaMalloc((void**)&GPUarray, TOTAL * sizeof(uint8_t));
    cudaMemcpy(GPUarray, image_array, TOTAL * sizeof(uint8_t), cudaMemcpyHostToDevice);
    inverseImageNoStriding << <blocksPerGrid, threadsPerBlock >> > (GPUarray);

    for (int i = 0; i < 11; i++) {
        int threads = pow(2, i);
        blocksPerGrid = TOTAL / threadsPerBlock;
        //cout << blocksPerGrid << "blockper \n";
        int warps = ceilf((float)threads / 32) * blocksPerGrid;
        //cout << "Warps: " << warps << "\n";
        const auto start = std::chrono::steady_clock::now();
        inverseImageNoStriding << <blocksPerGrid, threadsPerBlock >> > (GPUarray);

        const auto end = std::chrono::steady_clock::now();
        const std::chrono::duration<double, milli> elapsed_seconds{ end - start };
        //cout << "No striding" << "Amount of threads: " << threads << ", time: " << elapsed_seconds.count() << "\n";
        cout << elapsed_seconds.count() << "\n";
    }
    //inverseImageNoStriding << <blocksPerGrid, threadsPerBlock >> > (GPUarray);


    cudaMemcpy(new_image_array, GPUarray, TOTAL * sizeof(uint8_t), cudaMemcpyDeviceToHost);
    cudaFree(GPUarray);

    save_image_array(new_image_array);
}

void striding() {
    uint8_t* image_array = get_image_array();
    uint8_t* new_image_array = (uint8_t*)malloc(TOTAL);

    int threadsPerBlock = 1024 / 4;
    int blocksPerGrid = TOTAL / threadsPerBlock;


    uint8_t* GPUarray = NULL;
    cudaMalloc((void**)&GPUarray, TOTAL * sizeof(uint8_t));
    cudaMemcpy(GPUarray, image_array, TOTAL * sizeof(uint8_t), cudaMemcpyHostToDevice);

    inverseImageStriding << <1, threadsPerBlock >> > (GPUarray, threadsPerBlock);
    cout << "Striding \n";
    for (int i = 0; i < 11; i++) {
        int threads = pow(2, i);
        int warps = ceilf((float)threads / 32) * blocksPerGrid;
        cout << "Warps: " << warps << "\n";
        const auto start = std::chrono::steady_clock::now();
        inverseImageStriding << <1, threads >> > (GPUarray, threads);
        const auto end = std::chrono::steady_clock::now();
        const std::chrono::duration<double, milli> elapsed_seconds{ end - start };
        //cout << "Striding" << "Amount of threads: " << threads << ", time: " << elapsed_seconds.count() << "\n";
    }


    cudaMemcpy(new_image_array, GPUarray, TOTAL * sizeof(uint8_t), cudaMemcpyDeviceToHost);
    cudaFree(GPUarray);

    save_image_array(new_image_array);
}

void rComponent() {
    uint8_t* image_array = get_image_array();
    uint8_t* new_image_array = (uint8_t*)malloc(TOTAL);

    int threadsPerBlock = 1024 / 4;
    int blocksPerGrid = TOTAL / threadsPerBlock;
    int warps = ceil(threadsPerBlock / 32) * blocksPerGrid;

    uint8_t* GPUarray = NULL;
    cudaMalloc((void**)&GPUarray, TOTAL * sizeof(uint8_t));
    cudaMemcpy(GPUarray, image_array, TOTAL * sizeof(uint8_t), cudaMemcpyHostToDevice);

    inverseImageRComponent << <1, threadsPerBlock >> > (GPUarray, threadsPerBlock);

    for (int i = 0; i < 11; i++) {
        int threads = pow(2, i);
        const auto start = std::chrono::steady_clock::now();
        inverseImageRComponent << <1, threads >> > (GPUarray, threads);
        const auto end = std::chrono::steady_clock::now();
        const std::chrono::duration<double, milli> elapsed_seconds{ end - start };
        cout << "R-component" << "Amount of threads: " << threads << ", time: " << elapsed_seconds.count() << "\n";
    }


    cudaMemcpy(new_image_array, GPUarray, TOTAL * sizeof(uint8_t), cudaMemcpyDeviceToHost);
    cudaFree(GPUarray);

    save_image_array(new_image_array);
}

void reformattedImage() {
    uint8_t* image_array = get_image_array();
    uint8_t* new_image_array = (uint8_t*)malloc(TOTAL);
    uint8_t* reformattedInput = (uint8_t*)malloc(TOTAL);

    for (int i = 0; i < M * N; i++) {
        reformattedInput[i] = image_array[i*3];
        reformattedInput[(M * N) + i] = image_array[i*3 + 1];
        reformattedInput[(M * N * 2) + i] = image_array[i*3 + 2];
    }
    int threadsPerBlock = 1024 / 4;
    int blocksPerGrid = TOTAL / threadsPerBlock;
    int warps = ceil(threadsPerBlock / 32) * blocksPerGrid;

    uint8_t* GPUarray = NULL;
    cudaMalloc((void**)&GPUarray, TOTAL * sizeof(uint8_t));
    cudaMemcpy(GPUarray, reformattedInput, TOTAL * sizeof(uint8_t), cudaMemcpyHostToDevice);

    inverseImageColors << <1, threadsPerBlock >> > (GPUarray, threadsPerBlock);
    
    for (int i = 0; i < 11; i++) {
        int threads = pow(2, i);
        const auto start = std::chrono::steady_clock::now();
        inverseImageColors << <1, threads >> > (GPUarray, threads);
        const auto end = std::chrono::steady_clock::now();
        const std::chrono::duration<double, milli> elapsed_seconds{ end - start };
        cout << "Reformatted " << "Amount of threads: " << threads << ", time: " << elapsed_seconds.count() << "\n";
    }

    cudaMemcpy(reformattedInput, GPUarray, TOTAL * sizeof(uint8_t), cudaMemcpyDeviceToHost);
    cudaFree(GPUarray);
    for (int i = 0; i < M * N; i++) {
        new_image_array[i*3] = reformattedInput[i];
        new_image_array[i*3 + 1] = reformattedInput[i + (M * N)];
        new_image_array[i*3 + 2] = reformattedInput[i + (M * N)*2];
    }

    save_image_array(new_image_array);
}

void CPU() {
    uint8_t* image_array = get_image_array();
    
    for (int j = 0; j < 5; j++) {


        const auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < TOTAL; i++) {
            if (i % 3 == 0) {
                image_array[i] = (image_array[i] % 25) * 10;

            }
            else {
                image_array[i] = 255 - image_array[i];

            }
        }
        const auto end = std::chrono::steady_clock::now();
        const std::chrono::duration<double, milli> elapsed_seconds{ end - start };
        cout << "CPU " << "time: " << elapsed_seconds.count() << "\n";
    }
    save_image_array(image_array);
}



int main(void) {
    //noStriding();
    //striding();
    //rComponent();
    //reformattedImage();
    CPU();

    return 0;
}
