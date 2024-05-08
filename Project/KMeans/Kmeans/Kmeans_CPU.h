#include "kernel.cu"
class Kmeans_CPU {
	public :
		Kmeans_CPU(vector<Location>* locations, int iterations, int k, vector<Location>* centers);

		~Kmeans_CPU();
		void run(int max_iter);
		void print_centroids();
};