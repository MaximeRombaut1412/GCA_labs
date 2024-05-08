#include <ctime>
#include <fstream>
#include <iostream>  
#include <sstream>
#include <vector>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iomanip>
#include <chrono>
using namespace std;
//https://reasonabledeviations.com/2019/10/02/k-means-in-cpp/#c-preambles
#define K 4000
#define FILENAME "locations_50000.csv"
#define AMOUNT_OF_LOCATIONS 50000
struct LocationConstant {
    float lat, lon;
    int cluster;
    float minDistance;
};


struct Location {
    float lat, lon; //centerLat, centerLon;
    int cluster;
    float minDistance;

    //Constructors
    Location() : 
        lat(0.0),
        lon(0.0),
        cluster(-1),
        minDistance(FLT_MAX) {}

    Location(float lat, float lon) :
        lat(lat),
        lon(lon),
        cluster(-1),
        minDistance(FLT_MAX) {}
    Location(float lat, float lon, int cluster) :
        lat(lat),
        lon(lon),
        cluster(cluster),
        minDistance(FLT_MAX) {}
};

__constant__ LocationConstant constantCenters[K];
//__constant__ LocationConstant constantLocations[AMOUNT_OF_LOCATIONS];
// Location constantLocations[100000];

__global__ void SumKernel(Location* locations, Location* centers, int amount_of_locations, int amount_of_centers) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i < amount_of_locations) {
		for (int j = 0; j < amount_of_centers; j++) {
			float distance = sqrt((locations[i].lat - centers[j].lat) * (locations[i].lat - centers[j].lat) + (locations[i].lon - centers[j].lon) * (locations[i].lon - centers[j].lon));
			if (distance < locations[i].minDistance) {
				locations[i].minDistance = distance;
				locations[i].cluster = j;
			}
		}
	}
}

__global__ void AssignLocationKernel(Location* locations, Location* centers, int amount_of_locations, int amount_of_centers) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i < amount_of_locations) {
		for (int j = 0; j < amount_of_centers; j++) {
			float distance = sqrt((locations[i].lat - centers[j].lat) * (locations[i].lat - centers[j].lat) + (locations[i].lon - centers[j].lon) * (locations[i].lon - centers[j].lon));
			if (distance < locations[i].minDistance) {
				locations[i].minDistance = distance;
				locations[i].cluster = j;
			}
		}
	}
}

__global__ void AssignLocationKernelConstant(LocationConstant* locations, int amount_of_locations, int amount_of_centers) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < amount_of_locations) {
        int limit = min(amount_of_centers, K); // Ensure we don't exceed the size of constantCenters
        for (int j = 0; j < limit; j++) {
            float distance = sqrt((locations[i].lat - constantCenters[j].lat) * (locations[i].lat - constantCenters[j].lat) + (locations[i].lon - constantCenters[j].lon) * (locations[i].lon - constantCenters[j].lon));
            if (distance < locations[i].minDistance) {
                locations[i].minDistance = distance;
                locations[i].cluster = j;
            }
        }
    }
}
__global__ void CalculateCenterSumsKernel2(Location* locations, Location* centers, int amount_of_locations, int* centers_are_same, int amount_of_threads) {
    //VERY IMPORTANT TO INITIALIZE SHARED MEMORY
    __shared__ float sharedData[3072];
    // Initialize shared memory
    int index = threadIdx.x;
    int clusterIndex = blockIdx.x;
    int iterations = amount_of_locations / amount_of_threads;
    int remainder = amount_of_locations % amount_of_threads;
    int shared_mem_index = threadIdx.x * 3;

    for (int i = 0; i < 3; i++) {
        sharedData[shared_mem_index] = 0.0f;
        sharedData[shared_mem_index + 1] = 0.0f;
        sharedData[shared_mem_index + 2] = 0.0f;
	}
    __syncthreads();
    for (int i = 0; i < iterations; i++) {
        int locationIndex = index * iterations + i;
        if (locationIndex < amount_of_locations){
            if (locations[locationIndex].cluster == clusterIndex) {
                sharedData[shared_mem_index] += locations[locationIndex].lat;
                sharedData[shared_mem_index +1] += locations[locationIndex].lon;
                sharedData[shared_mem_index + 2] += 1;
            }
        }
    }
    // Handle the remainder
    if (index < remainder) {
        int locationIndex = amount_of_locations - remainder + index;
        if (locations[locationIndex].cluster == clusterIndex) {
            sharedData[shared_mem_index] += locations[locationIndex].lat;
            sharedData[shared_mem_index + 1] += locations[locationIndex].lon;
            sharedData[shared_mem_index + 2] += 1;
        }
    }
    __syncthreads();

    
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride && threadIdx.x + stride < blockDim.x) {
            // Add lat values
            sharedData[shared_mem_index] += sharedData[shared_mem_index + stride * 3];
            // Add lon values
            sharedData[shared_mem_index + 1] += sharedData[shared_mem_index + 1 + stride * 3];
            // Add count values
            sharedData[shared_mem_index + 2] += sharedData[shared_mem_index + 2 + stride * 3];
        }
        __syncthreads();
    }

    // After reduction, thread 0 stores the final sums in centers
    if (threadIdx.x == 0) {
        /*for (int i = 1; i < 1024; i++) {
			sharedData[0] += sharedData[i * 3];
			sharedData[1] += sharedData[i * 3 + 1];
			sharedData[2] += sharedData[i * 3 + 2];
		}*/


        float originalLat = centers[clusterIndex].lat;
        float originalLon = centers[clusterIndex].lon;
        //printf("Cluster: %d, lat: %f, lon: %f, count: %f\n", clusterIndex, sharedData[shared_mem_index], sharedData[shared_mem_index + 1], sharedData[shared_mem_index + 2]);
        if (sharedData[shared_mem_index + 2] > 0) {
            centers[clusterIndex].lat = sharedData[shared_mem_index] / sharedData[shared_mem_index + 2];
            centers[clusterIndex].lon = sharedData[shared_mem_index + 1] / sharedData[shared_mem_index + 2];
        }
        else {
            centers_are_same[clusterIndex] = 1;
		}
        if (originalLat == centers[clusterIndex].lat && originalLon == centers[clusterIndex].lon) {
			centers_are_same[clusterIndex] = 1;
		}
        centers[clusterIndex].minDistance = sharedData[shared_mem_index + 2];
    }
}
//
//__global__ void CalculateCenterSumsKernelConstant(LocationConstant* centers, int amount_of_locations, int* centers_are_same, int amount_of_threads) {
//    //VERY IMPORTANT TO INITIALIZE SHARED MEMORY
//    __shared__ float sharedData[3072];
//    // Initialize shared memory
//    int index = threadIdx.x;
//    int clusterIndex = blockIdx.x;
//    int iterations = amount_of_locations / amount_of_threads;
//    int remainder = amount_of_locations % amount_of_threads;
//    int shared_mem_index = threadIdx.x * 3;
//
//    for (int i = 0; i < 3; i++) {
//        sharedData[shared_mem_index] = 0.0f;
//        sharedData[shared_mem_index + 1] = 0.0f;
//        sharedData[shared_mem_index + 2] = 0.0f;
//    }
//    __syncthreads();
//    for (int i = 0; i < iterations; i++) {
//        int locationIndex = index * iterations + i;
//        if (locationIndex < amount_of_locations) {
//            if (constantLocations[locationIndex].cluster == clusterIndex) {
//                sharedData[shared_mem_index] += constantLocations[locationIndex].lat;
//                sharedData[shared_mem_index + 1] += constantLocations[locationIndex].lon;
//                sharedData[shared_mem_index + 2] += 1;
//            }
//        }
//    }
//    // Handle the remainder
//    if (index < remainder) {
//        int locationIndex = amount_of_locations - remainder + index;
//        if (constantLocations[locationIndex].cluster == clusterIndex) {
//            sharedData[shared_mem_index] += constantLocations[locationIndex].lat;
//            sharedData[shared_mem_index + 1] += constantLocations[locationIndex].lon;
//            sharedData[shared_mem_index + 2] += 1;
//        }
//    }
//    __syncthreads();
//
//
//    //for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
//    //    if (threadIdx.x < stride && threadIdx.x + stride < blockDim.x) {
//    //        // Add lat values
//    //        sharedData[shared_mem_index] += sharedData[shared_mem_index + stride * 3];
//    //        // Add lon values
//    //        sharedData[shared_mem_index + 1] += sharedData[shared_mem_index + 1 + stride * 3];
//    //        // Add count values
//    //        sharedData[shared_mem_index + 2] += sharedData[shared_mem_index + 2 + stride * 3];
//    //    }
//    //    __syncthreads();
//    //}
//
//    // After reduction, thread 0 stores the final sums in centers
//    if (threadIdx.x == 0) {
//        for (int i = 1; i < 1024; i++) {
//            sharedData[0] += sharedData[i * 3];
//            sharedData[1] += sharedData[i * 3 + 1];
//            sharedData[2] += sharedData[i * 3 + 2];
//        }
//
//
//        float originalLat = centers[clusterIndex].lat;
//        float originalLon = centers[clusterIndex].lon;
//        //printf("Cluster: %d, lat: %f, lon: %f, count: %f\n", clusterIndex, sharedData[shared_mem_index], sharedData[shared_mem_index + 1], sharedData[shared_mem_index + 2]);
//        if (sharedData[shared_mem_index + 2] > 0) {
//            centers[clusterIndex].lat = sharedData[shared_mem_index] / sharedData[shared_mem_index + 2];
//            centers[clusterIndex].lon = sharedData[shared_mem_index + 1] / sharedData[shared_mem_index + 2];
//        }
//        else {
//            centers_are_same[clusterIndex] = 1;
//        }
//        if (originalLat == centers[clusterIndex].lat && originalLon == centers[clusterIndex].lon) {
//            centers_are_same[clusterIndex] = 1;
//        }
//        centers[clusterIndex].minDistance = sharedData[shared_mem_index + 2];
//    }
//}

std::ostream& operator<<(std::ostream& os, const Location& location) {
    os << "(" << location.lat << ", " << location.lon << ")";
    return os;
}

float calculateDistanceCPU(Location a, Location b) {
    return sqrt((a.lat - b.lat) * (a.lat - b.lat) + (a.lon - b.lon) * (a.lon - b.lon));
}

//Read locations out of csv file and put them in vector
vector<Location> readLocationsCsv() {
    vector<Location> locations;
    string line;
    ifstream inputfile(FILENAME);
    while (getline(inputfile, line)) {
        stringstream lineStream(line);
        string bit;
        float lat, lon;
        getline(lineStream, bit, ',');
        lat = stof(bit);
        getline(lineStream, bit, '\n');
        lon = stof(bit);
        locations.push_back(Location(lat, lon));
    }
    return locations;
}
//Read locations out of csv file and put them in vector
vector<LocationConstant> readLocationsConstantCsv() {
    vector<LocationConstant> locations;
    string line;
    ifstream inputfile(FILENAME);
    while (getline(inputfile, line)) {
        stringstream lineStream(line);
        string bit;
        float lat, lon;
        getline(lineStream, bit, ',');
        lat = stof(bit);
        getline(lineStream, bit, '\n');
        lon = stof(bit);
        LocationConstant loc;
        loc.lat = lat;
        loc.lon = lon;
        loc.cluster = -1;
        loc.minDistance = FLT_MAX;
        locations.push_back(loc);
    }
    return locations;
}
void AssignLocationToCenter(Location* l, vector<Location>* centers) {
    for (int i = 0; i <centers->size(); i++) {
        float distance = calculateDistanceCPU(*l, centers->at(i));
        if (distance < l->minDistance) {
            l->minDistance = distance;
            l->cluster = i;
        }
    }
}

void AssignLocationToCenterGPU(vector<Location>* locations, vector<Location>* centers) {
    //Allocate memory on GPU
    Location* GPULocations = NULL;
    Location* GPUCenters = NULL;
    cudaMalloc((void**)&GPULocations, locations->size()  *sizeof(Location));
    cudaMalloc((void**)&GPUCenters, centers->size() * sizeof(Location));

    //Copy data from host to device
    cudaMemcpy(GPULocations, locations->data(), locations->size() * sizeof(Location), cudaMemcpyHostToDevice);
    cudaMemcpy(GPUCenters, centers->data(), centers->size() * sizeof(Location), cudaMemcpyHostToDevice);
    
    //Threads per block
    int threadsPerBlock = 1024;
    int blocksPerGrid = (locations->size() + threadsPerBlock - 1) / threadsPerBlock; //Can we optimize this?

    //Call kernel
    AssignLocationKernel<<<blocksPerGrid, threadsPerBlock>>>(GPULocations, GPUCenters, locations->size(), centers->size());

    //Copy data from device to host
    cudaMemcpy(locations->data(), GPULocations, locations->size() * sizeof(Location), cudaMemcpyDeviceToHost);
   

    //Free memory
    cudaFree(GPULocations);
    cudaFree(GPUCenters);
}
//Calculate the sum of all lat and lon values of the locations that belong to a cluster in the kernel
bool AssignLocationToCenterGPUS(vector<Location>* locations, vector<Location>* centers) {
    //Allocate memory on GPU
    vector<int> done(centers->size(), 0);

    int* GPUBools = NULL;
    Location* GPULocations = NULL;
    Location* GPUCenters = NULL;

    cudaMalloc((void**)&GPULocations, locations->size() * sizeof(Location));
    cudaMalloc((void**)&GPUCenters, centers->size() * sizeof(Location));
    cudaMalloc((void**)&GPUBools, centers->size() * sizeof(int));

    //Copy data from host to device
    cudaMemcpyAsync(GPULocations, locations->data(), locations->size() * sizeof(Location), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(GPUCenters, centers->data(), centers->size() * sizeof(Location), cudaMemcpyHostToDevice);
    //cudaMemcpyToSymbolAsync(constantLocations, locations->data(), locations->size() * sizeof(Location), 0, cudaMemcpyHostToDevice);



    //Threads per block
    int threadsPerBlock = 1024;
    int blocksPerGrid = (locations->size() + threadsPerBlock - 1) / threadsPerBlock; //Can we optimize this?

    //Call kernel
    AssignLocationKernel << <blocksPerGrid, threadsPerBlock >> > (GPULocations, GPUCenters, locations->size(), centers->size());
    cudaDeviceSynchronize();
    CalculateCenterSumsKernel2 << <centers->size(), 1024 >> > (GPULocations, GPUCenters, locations->size(), GPUBools, 1024);
    cudaDeviceSynchronize();
    //Copy data from device to host
    cudaMemcpy(locations->data(), GPULocations, locations->size() * sizeof(Location), cudaMemcpyDeviceToHost);
    cudaMemcpy(centers->data(), GPUCenters, centers->size() * sizeof(Location), cudaMemcpyDeviceToHost);
    cudaMemcpy(done.data(), GPUBools, centers->size() * sizeof(int), cudaMemcpyDeviceToHost);
    //Free memory
    cudaFree(GPULocations);
    cudaFree(GPUCenters);
    cudaFree(GPUBools);
    int count = 0;
    float test = 0;
    for (int i = 0; i < centers->size(); i++) {
		if (done[i]) {
			count++;
		}
        test += centers->at(i).minDistance;
	}
	if (count == centers->size()) {

        return true;
	}
    return false;
}

//
//bool AssignLocationAndCalculateSums(vector<LocationConstant>* locations, vector<LocationConstant>* centers) {
//    
//    //Allocate memory on GPU
//    LocationConstant* GPULocations = NULL;
//    LocationConstant* GPUCenters = NULL;
//    vector<int> done(centers->size(), 0);
//    int* GPUBools = NULL;
//
//   
//    //The centers only need to be read. By using constant memory we try to speed up the process
//    cudaMemcpyToSymbolAsync(constantCenters, centers->data(), centers->size() * sizeof(LocationConstant), 0, cudaMemcpyHostToDevice);
//    cudaMalloc((void**)&GPULocations, locations->size() * sizeof(LocationConstant));
//    //Copy data from host to device
//    cudaMemcpy(GPULocations, locations->data(), locations->size() * sizeof(Location), cudaMemcpyHostToDevice);
//
//    //Threads per block
//    int threadsPerBlock = 1024;
//    int blocksPerGrid = (locations->size() + threadsPerBlock - 1) / threadsPerBlock; //Can we optimize this?
//    cudaDeviceSynchronize();
//    //Call kernel
//    AssignLocationKernelConstant << <blocksPerGrid, threadsPerBlock >> > (GPULocations, locations->size(),centers->size());
//    
//    //Copy data from device to host
//    cudaMemcpy(locations->data(), GPULocations, locations->size() * sizeof(Location), cudaMemcpyDeviceToHost);
//    
//    //In the calculation of the new centers, the Locations need to be read a lot. We will try to make this faster by using constant memory.
//    cudaMemcpyToSymbolAsync(constantLocations, locations->data(), locations->size() * sizeof(LocationConstant), 0, cudaMemcpyHostToDevice);
//    cudaMalloc((void**)&GPUCenters, centers->size() * sizeof(LocationConstant));
//    cudaMemcpy(GPUCenters, centers->data(), centers->size() * sizeof(LocationConstant), cudaMemcpyHostToDevice);
//    CalculateCenterSumsKernelConstant << <centers->size(), 1024 >> > (GPUCenters, locations->size(), GPUBools, 1024);
//    cudaDeviceSynchronize();
//
//    //Free memory
//    cudaFree(GPULocations);
//    cudaFree(GPUCenters);
//    cudaFree(GPUBools);
//    int count = 0;
//    float test = 0;
//    for (int i = 0; i < centers->size(); i++) {
//        if (done[i]) {
//            count++;
//        }
//        test += centers->at(i).minDistance;
//    }
//    if (count == centers->size()) {
//        return true;
//    }
//    return false;
//}

void AssignLocationToCenterGPUConstant(vector<LocationConstant>* locations, vector<LocationConstant>* centers) {
    //Allocate memory on GPU
    LocationConstant* GPULocations = NULL;

    //The centers only need to be read. By using constant memory we try to speed up the process
    cudaMemcpyToSymbolAsync(constantCenters, centers->data(), centers->size() * sizeof(LocationConstant), 0, cudaMemcpyHostToDevice);
    cudaMalloc((void**)&GPULocations, locations->size() * sizeof(LocationConstant));

    //Copy data from host to device
    cudaMemcpy(GPULocations, locations->data(), locations->size() * sizeof(Location), cudaMemcpyHostToDevice);

    //Threads per block
    int threadsPerBlock = 1024;
    int blocksPerGrid = (locations->size() + threadsPerBlock - 1) / threadsPerBlock; //Can we optimize this?
    //cudaDeviceSynchronize();
    //Call kernel
    AssignLocationKernelConstant << <blocksPerGrid, threadsPerBlock >> > (GPULocations, locations->size(), centers->size());


    //Copy data from device to host
    cudaMemcpy(locations->data(), GPULocations, locations->size() * sizeof(Location), cudaMemcpyDeviceToHost);


    //Free memory
    cudaFree(GPULocations);
    //cudaFree(constantCenters);
}

void WriteCentersToFile(vector<Location>* centers, int k) {
	std::ofstream myfile;
	myfile.open("output.csv", std::ios::app);
	myfile << "============================" << endl;
	myfile << "K: " << k << endl;
	myfile << "----------------------------" << endl;
	for (vector<Location>::iterator it = centers->begin();
		it != centers->end(); ++it) {
		myfile << "Cluster: " << it->cluster <<", latitude:  " << std::fixed << std::setprecision(6) << it->lat << ", longitude: " << std::fixed << std::setprecision(6) <<it->lon << endl;
	}
	myfile << "----------------------------" << endl;
	myfile.close();
}
void WriteCentersToFileGPU(vector<Location>* centers, int k) {
    std::ofstream myfile;
    myfile.open("outputGPU.csv", std::ios::app);
    myfile << "============================" << endl;
    myfile << "K: " << k << endl;
    myfile << "----------------------------" << endl;
    for (vector<Location>::iterator it = centers->begin();
        it != centers->end(); ++it) {
        myfile << "Cluster: " << it->cluster << ", latitude:  " << std::fixed << std::setprecision(6) << it->lat << ", longitude: " << std::fixed << std::setprecision(6) << it->lon << endl;
    }
    myfile << "----------------------------" << endl;
    myfile.close();
}
void WriteCentersToFileGPUConstant(vector<LocationConstant>* centers, int k) {
    std::ofstream myfile;
    myfile.open("outputGPUConstant.csv", std::ios::app);
    myfile << "============================" << endl;
    myfile << "K: " << k << endl;
    myfile << "----------------------------" << endl;
    for (vector<LocationConstant>::iterator it = centers->begin();
        it != centers->end(); ++it) {
        myfile << "Cluster: " << it->cluster << ", latitude:  " << std::fixed << std::setprecision(6) << it->lat << ", longitude: " << std::fixed << std::setprecision(6) << it->lon << endl;
    }
    myfile << "----------------------------" << endl;
    myfile.close();
}
void WriteCentersToFileGPUAsync(vector<Location>* centers, int k) {
    std::ofstream myfile;
    myfile.open("outputGPUAsync.csv", std::ios::app);
    myfile << "============================" << endl;
    myfile << "K: " << k << endl;
    myfile << "----------------------------" << endl;
    for (vector<Location>::iterator it = centers->begin();
        it != centers->end(); ++it) {
        myfile << "Cluster: " << it->cluster << ", latitude:  " << std::fixed << std::setprecision(6) << it->lat << ", longitude: " << std::fixed << std::setprecision(6) << it->lon << endl;
    }
    myfile << "----------------------------" << endl;
    myfile.close();
}
void resetOutputFiles() {
    std::ofstream outputFile;
    outputFile.open("output.csv", std::ios::trunc); // Open file in truncation mode
    outputFile.close(); // Close the file
    std::cout << "Cpu file reset successfully." << std::endl;
    std::ofstream outputFileGPU;
    outputFileGPU.open("outputGPU.csv", std::ios::trunc); // Open file in truncation mode
    outputFileGPU.close(); // Close the file
    std::cout << "Gpu file reset successfully." << std::endl;

    std::ofstream outputFileGPUConstant;
    outputFileGPUConstant.open("outputGPUConstant.csv", std::ios::trunc); // Open file in truncation mode
    outputFileGPUConstant.close(); // Close the file
    std::cout << "Gpu constant file reset successfully." << std::endl;

    std::ofstream outputFileGPUAsync;
    outputFileGPUAsync.open("outputGPUAsync.csv", std::ios::trunc); // Open file in truncation mode
    outputFileGPUAsync.close(); // Close the file
    std::cout << "Gpu Async file reset successfully." << std::endl;
}
bool calculateCentersumsGPU(vector<Location>* centers) {
    for (int i = 0; i < centers->size(); i++) {
		float lat = centers->at(i).lat / centers->at(i).minDistance;
		float lon = centers->at(i).lon / centers->at(i).minDistance;
		if (lat == centers->at(i).lat && lon == centers->at(i).lon) {
			return true;
		}
		(*centers)[i].lat = lat;
		(*centers)[i].lon = lon;
	}
}

bool CalculateCenterSums(vector<Location>* locations, vector<Location>* centers) {
    vector<float> lat_and_lon_sums; //Vector to add sums to. First element is the lat of the first center. The second element the lon of first. Third lat of second center,...
    vector<int> amount_of_locations; //The amount of locations per cluster. Needed to divide at the end
    for (int i = 0; i < centers->size(); i++) {
        lat_and_lon_sums.push_back(0);
        lat_and_lon_sums.push_back(0);
        amount_of_locations.push_back(0);
    }
    for (int i = 0; i < locations->size(); i++) {
        Location l = locations->at(i);
        amount_of_locations.at(l.cluster) += 1;
        int center = l.cluster *2;
        lat_and_lon_sums.at(center) += l.lat;
        lat_and_lon_sums.at(center + 1) += l.lon;
    }
    int clusters_with_same_values = 0;
    int notSame = 0;
    for (int i = 0; i < centers->size(); i++) {
        if (amount_of_locations.at(i) != 0) {
            float lat = lat_and_lon_sums.at(i * 2) / amount_of_locations.at(i);
            float lon = lat_and_lon_sums.at(i * 2 + 1) / amount_of_locations.at(i);
            if (lat == centers->at(i).lat && lon == centers->at(i).lon) {
                clusters_with_same_values++;
            }
            else {
                notSame = i;
            }
            (*centers)[i].lat = lat;
            (*centers)[i].lon = lon;
        }
        else {
            clusters_with_same_values++;
        }
    }
    if (clusters_with_same_values == centers->size()) {
		return true;
	}
    
    return false;
}

bool CalculateCenterSumsConstant(vector<LocationConstant>* locations, int k, vector<LocationConstant>* centers) {
    /*for (int i =0; i < K; i++) {
		LocationConstant l = constantCenters[i];
		cout << "Cluster: " << l.cluster << " lat: " << l.lat << " lon: " << l.lon << endl;
	}*/
    vector<float> lat_and_lon_sums; //Vector to add sums to. First element is the lat of the first center. The second element the lon of first. Third lat of second center,...
    vector<int> amount_of_locations; //The amount of locations per cluster. Needed to divide at the end
    for (int i = 0; i < k; i++) {
        lat_and_lon_sums.push_back(0);
        lat_and_lon_sums.push_back(0);
        amount_of_locations.push_back(0);
    }
    for (int i = 0; i < locations->size(); i++) {
        LocationConstant l = locations->at(i);
        amount_of_locations.at(l.cluster) += 1;
        int center = l.cluster * 2;
        lat_and_lon_sums.at(center) += l.lat;
        lat_and_lon_sums.at(center + 1) += l.lon;
    }
    int clusters_with_same_values = 0;
    int notSame = 0;
    for (int i = 0; i < centers->size(); i++) {
        if (amount_of_locations.at(i) != 0) {
            float lat = lat_and_lon_sums.at(i * 2) / amount_of_locations.at(i);
            float lon = lat_and_lon_sums.at(i * 2 + 1) / amount_of_locations.at(i);
            if (lat == centers->at(i).lat && lon == centers->at(i).lon) {
                clusters_with_same_values++;
            }
            else {
                notSame = i;
            }
            (*centers)[i].lat = lat;
            (*centers)[i].lon = lon;
        }
        else {
            clusters_with_same_values++;
        }
    }
    if (clusters_with_same_values == k) {
        return true;
    }

    return false;
}
//Here we calculate the clusters with the CPU.
//Paramters:
//  * locations: locations that need to be clustered
//  * iterations: amount of iterations before quiting
//  * k: amount of clusters that will be used
void KmeansCPU(vector<Location>* locations, int iterations, int k, vector<Location>* centers) {
    int amount_of_locations = locations->size();

    for (int j = 0; j < iterations; j++) {
        for (int i = 0; i < amount_of_locations; i++) {
            AssignLocationToCenter(&locations->at(i), centers);
        }
        if (CalculateCenterSums(locations, centers)) {
            WriteCentersToFile(centers,k);
            break;
        }
    }
}

void KmeansGPU(vector<Location>* locations, int iterations, int k, vector<Location>* centers) {
	for (int j = 0; j < iterations; j++) {
        AssignLocationToCenterGPU(locations, centers);
		if (CalculateCenterSums(locations, centers)) {
			WriteCentersToFileGPU(centers, k);
			break;
		}
	}
}
void KmeansGPUConstant(vector<LocationConstant>* locations, int iterations, int k, vector<LocationConstant>* centers) {
    for (int j = 0; j < iterations; j++) {
        AssignLocationToCenterGPUConstant(locations, centers);
        if (CalculateCenterSumsConstant(locations, k,centers)) {
            WriteCentersToFileGPUConstant(centers, k);
            break;
        }
    }
    
}
void KmeansGPUWithSum(vector<Location>* locations, int iterations, int k, vector<Location>* centers) {
    for (int j = 0; j < iterations; j++) {
        if (AssignLocationToCenterGPUS(locations, centers)) {
            WriteCentersToFileGPUAsync(centers, k);
            break;
        }
    }
}
//void KmeansGPUSumAndConstant(vector<LocationConstant>* locations, int iterations, int k, vector<LocationConstant>* centers) {
//	for (int j = 0; j < iterations; j++) {
//        for (int j = 0; j < iterations; j++) {
//            if (AssignLocationAndCalculateSums(locations, centers)) {
//                WriteCentersToFileGPUAsync(centers, k);
//                break;
//            }
//        }
//	}
//}


void ResetLocationClusters(vector<Location>* locations) {
	for (int i = 0; i < locations->size(); i++) {
		locations->at(i).cluster = -1;
		locations->at(i).minDistance = DBL_MAX;
	}
}
void ResetLocationClustersConstant(vector<LocationConstant>* locations) {
    for (int i = 0; i < locations->size(); i++) {
        locations->at(i).cluster = -1;
        locations->at(i).minDistance = DBL_MAX;
    }
}
vector<Location> GetCenterLocations(vector<Location>* locations, vector<int> indices) {
    vector<Location> centers;
    for (int i = 0; i < indices.size(); i++) {
        Location l = locations->at(indices.at(i));
        l.cluster = 0;
        centers.push_back(Location(l.lat, l.lon, i));
    }
    return centers;
}
vector<LocationConstant> GetCenterLocationsConstant(vector<LocationConstant>* locations, vector<int> indices) {
    vector<LocationConstant> centers;
    for (int i = 0; i < indices.size(); i++) {
        LocationConstant l = locations->at(indices.at(i));
        l.cluster = 0;

        LocationConstant center;
        center.lat = l.lat;
        center.lon = l.lon;
        center.cluster = i;
        centers.push_back(center);
    }
    return centers;
}
vector<int> getCenterpointIndices(int amount_of_locations, int k) {
    vector<int> indices;
    for (int i = 0; i < k; i++) {
		    indices.push_back(rand() % amount_of_locations);
	}
	return indices;
}
int main()
{
    resetOutputFiles();
    vector<Location> locations = readLocationsCsv(); //Get locations
    vector<Location> locations2 = readLocationsCsv();
    vector<LocationConstant> locations_constant = readLocationsConstantCsv();
    //vector<LocationConstant> locations_constant2 = readLocationsConstantCsv();
    vector<Location> locations3 = readLocationsCsv();

    for (int i = 0; i < 5; i++) {

        vector<int> indices = getCenterpointIndices(locations.size(), K - i*2);

        vector<Location> centers = GetCenterLocations(&locations, indices);
        vector<Location> centers2 = GetCenterLocations(&locations2, indices);
        vector<Location> centers3 = GetCenterLocations(&locations3, indices);
        vector<LocationConstant> centers_constant = GetCenterLocationsConstant(&locations_constant, indices);
        //vector<LocationConstant> centers_constant2 = GetCenterLocationsConstant(&locations_constant2, indices);


        //Execute Kmeans on CPU
        const auto startCPU = std::chrono::steady_clock::now();
        KmeansCPU(&locations, 10000, K-i, &centers);
        const auto endCPU = std::chrono::steady_clock::now();
        cout << "CPU done" << endl;
        //ResetLocationClusters(&locations);
        //Execute Kmeans on GPU
        const auto startGPU = std::chrono::steady_clock::now();
        KmeansGPU(&locations2, 10000, K - i*2, &centers2);
        const auto endGPU = std::chrono::steady_clock::now();
        cout << "GPU done" << endl;
        //Execute Kmeans on GPU using constant memory
        const auto startGPUC = std::chrono::steady_clock::now();
        KmeansGPUConstant(&locations_constant, 10000, K - i*2, &centers_constant);
        const auto endGPUC = std::chrono::steady_clock::now();
        cout << "GPUC done" << endl;
        //Execute Kmeans on GPU with sum
        const auto startGPUS = std::chrono::steady_clock::now();
        KmeansGPUWithSum(&locations3, 10000, K - i*2, &centers3);
        const auto endGPUS = std::chrono::steady_clock::now();
        cout << "GPUS done" << endl;

        /*const auto startGPUConstantSum = std::chrono::steady_clock::now();
        KmeansGPUSumAndConstant(&locations_constant2, 10000, K - i * 2, &centers_constant2);
        const auto endGPUConstantSum = std::chrono::steady_clock::now();
        cout << "GPUConstantSum done" << endl;*/

        //Reset clusters
        ResetLocationClusters(&locations);
        ResetLocationClusters(&locations2);
        ResetLocationClustersConstant(&locations_constant);
        //ResetLocationClustersConstant(&locations_constant2);
        ResetLocationClusters(&locations3);
        //ResetLocationClustersConstant();
        const std::chrono::duration<double, milli> elapsed_secondsCPU{ endCPU - startCPU };
        const std::chrono::duration<double, milli> elapsed_secondsGPU{ endGPU - startGPU };
        const std::chrono::duration<double, milli> elapsed_secondsGPUC{ endGPUC - startGPUC };
        const std::chrono::duration<double, milli> elapsed_secondsGPUS{ endGPUS - startGPUS };
        //const std::chrono::duration<double, milli> elapsed_secondsGPUConstantSum{ endGPUConstantSum - startGPUConstantSum };
        cout << "CPU: " << elapsed_secondsCPU.count() << "ms" << endl;
        cout << "GPU: " << elapsed_secondsGPU.count() << "ms" << endl;
        cout << "GPU Constant: " << elapsed_secondsGPUC.count() << "ms" << endl;
        cout << "GPUS: " << elapsed_secondsGPUS.count() << "ms" << endl;
        //cout << "GPU Constant Sum: " << elapsed_secondsGPUConstantSum.count() << "ms" << endl;
        cout << endl;
        //cout << "K: " << k - i * 2 << endl;
    }
    //KmeansCPU(&locations, 20, 5);
    cout << "test";
    return 0;
}
