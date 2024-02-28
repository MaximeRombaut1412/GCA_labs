#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <cmath>
#include <chrono>
#include <random>
using namespace std;

//void addSequential(int* a, int numElements)
//{
//    for (int i = 1; i < numElements; i++)
//    {
//        a[0] += a[i];
//    }
//}

__global__ void maxDetectionAtomic(int* max, int* a)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    atomicMax(max, a[i]);
}

__global__ void max_reduce(int* maxValue, int* inputData, int dataSize,int offset) {
    int idx = threadIdx.x;
  
    if (idx < dataSize) {
        for (int s = 1; s < dataSize; s *= 2) {
            if (idx < dataSize / (2 * s)) {

                int l = inputData[(idx * 2)+ offset];
                int r = inputData[(idx * 2) + 1 + offset];
                inputData[idx+offset] = max(l, r);

            }
            __syncthreads();
        }
    }
    __syncthreads();
    if (idx == 0) {
        //printf("Offset: %d\n", offset);
        //printf("MaxVal: %d\n", inputData[offset]);
        if (inputData[offset] > *maxValue) {
            *maxValue = inputData[offset];
        }
    }
}




int main()
{
    using std::chrono::high_resolution_clock;
    using std::chrono::duration_cast;
    using std::chrono::milliseconds;



    srand(time(NULL));

    const int numElements = pow(2,25);
    int* h_input = (int*)malloc(numElements * sizeof(int));
    int* maxV = (int*)malloc(sizeof(int));

    int* h_input2 = (int*)malloc(numElements * sizeof(int));
    int* maxV2 = (int*)malloc(sizeof(int));
    *maxV = -1;
    *maxV2 = -1;


    std::random_device rd;
    std::mt19937 gen(rd());

    std::uniform_int_distribution<int> dis(1, numElements * 2);

    //int randomNumber = dis(gen);

    for (int i = 0; i < numElements; ++i)
    {
        //h_input[i] = rand() % (numElements * 10) + 1;
        int value = dis(gen);
        h_input[i] = value;
        h_input2[i] = value;
    }
    //h_input[2492] = 20000;
    auto sequentialStart = high_resolution_clock::now();
    int maxSequential = -1;

    //int firstMax = 0;
    //int secondMax = 0;


    //int test = 0;
    //std::vector<int> index;
    for (int i = 0; i < numElements; i++) {
        if (h_input[i] > maxSequential) {
            maxSequential = h_input[i];
            
            //test = i;
        }
        /*if (i < numElements / 2) {
            if (h_input[i] > firstMax) {
                firstMax = h_input[i];
            }
            
        }
        else {
            if (h_input[i] > secondMax) {
                secondMax = h_input[i];
            }
        }*/
    }
    //cout << "FirstMax: " << firstMax << " SecondMax: " << secondMax << "\n";
    //for (int i = 0; i < numElements; i++) {
    //    if (h_input[i] == maxSequential) {
    //        //maxSequential = h_input[i];
    //        index.push_back(i);
    //    }
    //}
    //for (int k = 0; k < index.size(); k++) {
    //    //cout << "Index: " << index[k] << "\n";
    //}
    //cout << "MaxIndex: " << index. << "\n";
    auto sequentialStop =  high_resolution_clock::now();
    std::chrono::duration<double, milli> sTime = sequentialStop - sequentialStart;
    cout << "max sequential: " << maxSequential << "; time: " << sTime.count() << endl;


    int* d_inputAtomic = NULL;
    cudaMalloc((void**)&d_inputAtomic, numElements * sizeof(int));

    int* d_maxAtomic = NULL;
    cudaMalloc((void**)&d_maxAtomic, sizeof(int));

   /* int* d_inputReduction = NULL;
    cudaMalloc((void**)&d_inputReduction, numElements * sizeof(int));
    int* d_maxReduction = NULL;
    cudaMalloc((void**)&d_maxReduction, sizeof(int));*/


    cudaMemcpy(d_inputAtomic, h_input, numElements * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_maxAtomic, maxV, sizeof(int), cudaMemcpyHostToDevice);

    //cudaMemcpy(d_inputReduction, h_input, numElements * sizeof(int), cudaMemcpyHostToDevice);
    //cudaMemcpy(d_maxReduction, maxV, sizeof(int), cudaMemcpyHostToDevice);


    
    int threadsPerBlock = 1024; //de threads per block is telkens de helft van het aantal input elements. Het maximum threads op 1 block is 1024. Dit betekent dus dat we hier maximum 2048 elementen kunnen hebben. Kunnen we dan indien er meerdere zijn meerdere blocks gebruiken?
    if (numElements < 1024) {
        threadsPerBlock = numElements;
    }
    int blocksPerGrid = (int)fmax(ceil(numElements/1024),1);

    auto aStrt = high_resolution_clock::now();
    maxDetectionAtomic << <blocksPerGrid, threadsPerBlock >> > (d_maxAtomic, d_inputAtomic);
    auto aStp = high_resolution_clock::now();
    std::chrono::duration<double, milli> aT =aStp - aStrt;
    cudaMemcpy(maxV, d_maxAtomic, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_input, d_inputAtomic, numElements * sizeof(int), cudaMemcpyDeviceToHost);
    cout << "max atomic: " << *maxV << "; time: " << aT.count() << endl;


    cudaFree(d_inputAtomic);
    cudaFree(d_maxAtomic);
    //int blocksPerGridReduce = (int)fmax(ceil((numElements/2) / 1024), 1);

    /*for (int i = 0; i < numElements; i++) {
        cout << " " << h_input2[i];
    }
    cout << "\n";*/

    int* d_inputReduction = NULL;
    cudaMalloc((void**)&d_inputReduction, numElements * sizeof(int));
    int* d_maxReduction = NULL;
    cudaMalloc((void**)&d_maxReduction, sizeof(int));
    
    cudaMemcpy(d_inputReduction, h_input2, numElements * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_maxReduction, maxV2, sizeof(int), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    auto rStrt = high_resolution_clock::now();
    
    int offset = 0;

    for (int i = 0; i < ceil(fmax((numElements / 2048), 1)); i++) {
        if (numElements - 2048 * i < 2048) {
            max_reduce << <1, numElements / 2 - 2048 * i >> > (d_maxReduction, d_inputReduction, numElements - 2048 * i,offset);
            //cudaDeviceSynchronize();
        }
        else {
            max_reduce << <1, 1024 >> > (d_maxReduction, d_inputReduction, 2048,offset);
            //cudaDeviceSynchronize();
        }
        offset += 2048;
    }

    /*for (int i = 0; i < numElements / 2048; i++) {
        max_reduce << <1, 1024 >> > (d_maxReduction, d_inputReduction, 2048, offset);
        cudaDeviceSynchronize();
        offset += 2048;
    }*/

    
    auto rStp = high_resolution_clock::now();
    std::chrono::duration<double, milli> rT =rStp - rStrt;

    cudaMemcpy(maxV2, d_maxReduction, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_input2, d_inputReduction, numElements * sizeof(int), cudaMemcpyDeviceToHost);

    cout << "max reduction: " << maxV2[0] << "; time: " << rT.count() << endl;

    cudaFree(d_inputReduction);
    cudaFree(d_maxReduction);

    free(h_input);
    free(maxV);
    free(h_input2);
    free(maxV2);

    return 0;
}
