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

struct Location {
    float lat, lon;
    int cluster;
    double minDistance;

    //Constructors
    Location() : 
        lat(0.0),
        lon(0.0),
        cluster(-1),
        minDistance(DBL_MAX) {}

    Location(float lat, float lon) :
        lat(lat),
        lon(lon),
        cluster(-1),
        minDistance(DBL_MAX) {}
    Location(float lat, float lon, int cluster) :
        lat(lat),
        lon(lon),
        cluster(cluster),
        minDistance(DBL_MAX) {}
};

__global__ void AssignLocationKernel(Location* locations, Location* centers, int amount_of_locations, int amount_of_centers) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i < amount_of_locations) {
		for (int j = 0; j < amount_of_centers; j++) {
			double distance = sqrt((locations[i].lat - centers[j].lat) * (locations[i].lat - centers[j].lat) + (locations[i].lon - centers[j].lon) * (locations[i].lon - centers[j].lon));
			if (distance < locations[i].minDistance) {
				locations[i].minDistance = distance;
				locations[i].cluster = j;
			}
		}
	}
}

//struct Center {
//    float lat, lon;
//    int cluster_number;
//    Center(float lat, float lon, int cluster_number) :
//        lat(lat),
//        lon(lon),
//        cluster_number(cluster_number){}
//};

std::ostream& operator<<(std::ostream& os, const Location& location) {
    os << "(" << location.lat << ", " << location.lon << ")";
    return os;
}

double calculateDistanceCPU(Location a, Location b) {
    return sqrt((a.lat - b.lat) * (a.lat - b.lat) + (a.lon - b.lon) * (a.lon - b.lon));
}

//Read locations out of csv file and put them in vector
vector<Location> readLocationsCsv() {
    vector<Location> locations;
    string line;
    ifstream inputfile("locations_50000.csv");
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
void AssignLocationToCenter(Location* l, vector<Location>* centers) {
    for (int i = 0; i <centers->size(); i++) {
        double distance = calculateDistanceCPU(*l, centers->at(i));
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


void resetOutputFiles() {
    std::ofstream outputFile;
    outputFile.open("output.csv", std::ios::trunc); // Open file in truncation mode
    outputFile.close(); // Close the file
    std::cout << "Cpu file reset successfully." << std::endl;
    std::ofstream outputFileGPU;
    outputFileGPU.open("outputGPU.csv", std::ios::trunc); // Open file in truncation mode
    outputFileGPU.close(); // Close the file
    std::cout << "Gpu file reset successfully." << std::endl;
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
    for (int i = 0; i < centers->size(); i++) {
        float lat = lat_and_lon_sums.at(i*2) / amount_of_locations.at(i);
        float lon = lat_and_lon_sums.at(i*2 + 1) / amount_of_locations.at(i);
        if(lat == centers->at(i).lat && lon == centers->at(i).lon){
            clusters_with_same_values++;
		}
        (*centers)[i].lat = lat;
        (*centers)[i].lon = lon;
    }
    if (clusters_with_same_values == centers->size()) {
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
    /*vector<Location> centers;
    int amount_of_locations = locations->size();
    srand(time(0));
    for (int i = 0; i < k; i++) {
        Location l = locations->at(rand() % amount_of_locations);
        centers.push_back(Location(l.lat,l.lon, i));
    }*/
    int amount_of_locations = locations->size();
    for (int j = 0; j < iterations; j++) {
        for (int i = 0; i < amount_of_locations; i++) {
            AssignLocationToCenter(&locations->at(i), centers);
        }
        if (CalculateCenterSums(locations, centers)) {
            WriteCentersToFile(centers,k);
            break;
        }
        /*else {
            WriteCentersToFile(&centers);
        }*/
    }
}

void KmeansGPU(vector<Location>* locations, int iterations, int k, vector<Location>* centers) {
	/*vector<Location> centers;
	int amount_of_locations = locations->size();
	srand(time(0));
	for (int i = 0; i < k; i++) {
		Location l = locations->at(rand() % amount_of_locations);
		centers.push_back(Location(l.lat, l.lon, i));
	}*/


	for (int j = 0; j < iterations; j++) {
        AssignLocationToCenterGPU(locations, centers);
		if (CalculateCenterSums(locations, centers)) {
			WriteCentersToFileGPU(centers, k);
			break;
		}
		/*else {
			WriteCentersToFile(&centers);
		}*/
	}
}



void ResetLocationClusters(vector<Location>* locations) {
	for (int i = 0; i < locations->size(); i++) {
		locations->at(i).cluster = -1;
		locations->at(i).minDistance = DBL_MAX;
	}
}

vector<Location> GetCenterLocations(vector<Location>* locations, int k) {
    vector<Location> centers;
    int amount_of_locations = locations->size();
    srand(time(0));
    for (int i = 0; i < k; i++) {
        Location l = locations->at(rand() % amount_of_locations);
        centers.push_back(Location(l.lat, l.lon, i));
    }
    return centers;
}


int main()
{

    resetOutputFiles();
    vector<Location> locations= readLocationsCsv(); //Get locations
    int k = 24;
    vector<Location> centers = GetCenterLocations(&locations, k);
    for (int i = 0; i < k/2; i++) {
        //Execute Kmeans on CPU
        const auto startCPU = std::chrono::steady_clock::now();
        KmeansCPU(&locations, 1000, k-i*2, &centers);
        const auto endCPU = std::chrono::steady_clock::now();
        //Execute Kmeans on GPU
        const auto startGPU = std::chrono::steady_clock::now();
        KmeansGPU(&locations, 1000, k - i * 2, &centers);
        const auto endGPU = std::chrono::steady_clock::now();
        //Reset clusters
        ResetLocationClusters(&locations);
        const std::chrono::duration<double, milli> elapsed_secondsCPU{ endCPU - startCPU };
        const std::chrono::duration<double, milli> elapsed_secondsGPU{ endGPU - startGPU };
        cout << "CPU: " << elapsed_secondsCPU.count() << "ms" << endl;
        cout << "GPU: " << elapsed_secondsGPU.count() << "ms" << endl;
        //cout << "K: " << k - i * 2 << endl;
    }
    //KmeansCPU(&locations, 20, 5);
    cout << "test";
    return 0;
}

//NOTES
//--------------
//Indien veel clusters zouden we de max parallel kunnen bereken?
//Kijken of het berekenen van een nieuw centerpunt in parallel kan -> De som van lat en lon waarden van alle clusters moeten opgeteld worden en dan gedeeld om zo een average te krijgen.
//Mogelijk om de waarden visueel voor te stellen in een grafiek
//Enkel hiërarchieën weergeven
//
