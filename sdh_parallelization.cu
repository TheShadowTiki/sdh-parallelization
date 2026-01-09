/* ==================================================================
	Programmer: Abdul-Malik Zekri (zekri2@usf.edu)
	The optimized parallelized SDH algorithm implementation for 3D data
   ==================================================================
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>


#define BOX_SIZE	23000 /* size of the data box on one dimension            */

/* descriptors for single atom in the tree */
typedef struct atom_list_desc {
	double *x_list;
	double *y_list;
	double *z_list;
} atom_list;

typedef struct hist_entry{
	//float min;
	//float max;
	long long d_cnt;   /* need a long long type as the count might be huge */
} bucket;


bucket * histogram;		/* list of all buckets in the histogram   */
long long	PDH_acnt;	/* total number of data points            */
int num_buckets;		/* total number of buckets in the histogram */
double   PDH_res;		/* value of w                             */
atom_list atoms;		/* list of all data points                */
int PDH_block_size;		/* block size or number of threads per block*/

/* These are for an old way of tracking time */
struct timezone Idunno;	
struct timeval startTime, endTime;


/* 
	distance of two points in the atom_list 
*/
__host__ __device__ double p2p_distance(int ind1, int ind2, atom_list atoms) {
	
	double x1 = atoms.x_list[ind1];
	double x2 = atoms.x_list[ind2];
	double y1 = atoms.y_list[ind1];
	double y2 = atoms.y_list[ind2];
	double z1 = atoms.z_list[ind1];
	double z2 = atoms.z_list[ind2];
		
	return sqrt((x1 - x2)*(x1-x2) + (y1 - y2)*(y1 - y2) + (z1 - z2)*(z1 - z2));
}

// Distance function used in kernel to utilize loading data directly to thread registers and shared memory.
__device__ double p2p_distance_device(double x1, double y1, double z1, double x2, double y2, double z2) {

	return sqrt((x1 - x2)*(x1-x2) + (y1 - y2)*(y1 - y2) + (z1 - z2)*(z1 - z2));
}


/* 
	brute-force SDH solution in a single CPU thread 
*/
int PDH_baseline() {
	int i, j, h_pos;
	double dist;
	
	for(i = 0; i < PDH_acnt; i++) {
		for(j = i+1; j < PDH_acnt; j++) {
			dist = p2p_distance(i, j, atoms);
			h_pos = (int) (dist / PDH_res);
			histogram[h_pos].d_cnt++;
		} 
	}
	return 0;
}


/* 
	Kernel for parallel SDH solution implementing tiling, output privatization, 
	intrablock thread balancing, and interblock block balancing.
*/

__global__ void SDH_kernel(atom_list atoms, bucket* histogram, int PDH_acnt, int PDH_res, int numBuckets) {

    // Shared memory for R_block tiles and private output histogram
    extern __shared__ char shared_mem[];
    double *R_block_x = (double*)shared_mem;
    double *R_block_y = (double*)&R_block_x[blockDim.x];
    double *R_block_z = (double*)&R_block_y[blockDim.x];
	int *shm_out = (int*)(R_block_z + blockDim.x);

	// Initializing variables for ease of use
    int threadID = blockDim.x * blockIdx.x + threadIdx.x;
    int blockSize = blockDim.x;
	int totalBlocks = gridDim.x;
	int activeThreads = (blockIdx.x == (totalBlocks - 1)) ? (PDH_acnt - blockDim.x*(totalBlocks-1)) : blockDim.x; // Calculates how many threads in the block are within the data bound
    double dist;
    int h_pos;

	// Initialize shm_out
	for (int t = threadIdx.x; t < numBuckets; t += blockDim.x)
	{
		shm_out[t] = 0;
	}
	__syncthreads();  // Ensure all threads have initialized shm_out

    if (threadID < PDH_acnt) // Ensure threads are within data bounds
    {
        // Load atom coordinates for this thread
        double x1 = atoms.x_list[threadID];
        double y1 = atoms.y_list[threadID];
        double z1 = atoms.z_list[threadID];

        // Interblock Distance Calculation (Between Blocks)
		int activeBlocks = totalBlocks;
		int blockLoopLength = totalBlocks/2 - ((blockIdx.x < totalBlocks/2) || (totalBlocks%2 == 1) ? 0 : 1); // Determines how many blocks this block needs to compare to
		int balancedBlockIndex;
        for (int i = 0; i < blockLoopLength; i++) {
			balancedBlockIndex = (blockIdx.x + i + 1) % activeBlocks; // Determines which block to compare to at this iteration
			
            // Load another block's atoms into shared memory if they are within bounds
			for (int t = threadIdx.x; t < blockDim.x; t += activeThreads)
			{
            	int R_threadID = blockDim.x * balancedBlockIndex + t;

				if (R_threadID < PDH_acnt) {
					R_block_x[t] = atoms.x_list[R_threadID];
					R_block_y[t] = atoms.y_list[R_threadID];
					R_block_z[t] = atoms.z_list[R_threadID];
				}
			}
            __syncthreads();  // Ensure all threads have loaded R_block before proceeding

            // Calculate distance between x1, y1, z1 and points in R_block
            for (int j = 0; j < blockSize && (blockDim.x * balancedBlockIndex + j) < PDH_acnt; j++) {
                dist = p2p_distance_device(x1, y1, z1, R_block_x[j], R_block_y[j], R_block_z[j]);
                h_pos = (int)(dist / PDH_res);
                atomicAdd((int*)&(shm_out[h_pos]), 1);
            }
            __syncthreads();  // Ensure all threads have completed updates before next iteration
        }

        // Intrablock Distance Calculation (Within the Same Block)
        // Load all points within this block to shared memory
        R_block_x[threadIdx.x] = x1;
        R_block_y[threadIdx.x] = y1;
        R_block_z[threadIdx.x] = z1;
        __syncthreads();  // Ensure all threads have loaded their data

		int threadLoopLength = activeThreads/2 - ((threadIdx.x < activeThreads/2) || (activeThreads%2 == 1) ? 0 : 1); // Determines how many threads this thread needs to compare to
		int balancedIndex;
		for (int j = 0; j < threadLoopLength; j++) {
			balancedIndex = (threadIdx.x + j + 1) % activeThreads; // Determines which thread to compare to at this iteration
            dist = p2p_distance_device(x1, y1, z1, R_block_x[balancedIndex], R_block_y[balancedIndex], R_block_z[balancedIndex]);
            h_pos = (int)(dist / PDH_res);
            atomicAdd((int*)&(shm_out[h_pos]), 1);
        }

    }
	__syncthreads();  // Ensure all threads have completed updates

	// Update final output histogram with private copy
	if (threadIdx.x == 0)
	{
		for (int a = 0; a < numBuckets; a++)
		{
			atomicAdd((int*)&(histogram[a].d_cnt), shm_out[a]);
		}
	}

}


/* 
	host function for parallel SDH solution over PDH_acnt GPU threads
*/

int PDH_parallel() {

	// Allocate device (GPU) memory
	atom_list atoms_d;
	bucket *histogram_d;
	cudaMalloc((void**) &(atoms_d.x_list), PDH_acnt*sizeof(double));
	cudaMalloc((void**) &(atoms_d.y_list), PDH_acnt*sizeof(double));
	cudaMalloc((void**) &(atoms_d.z_list), PDH_acnt*sizeof(double));

	cudaMalloc((void**) &histogram_d, num_buckets*sizeof(bucket));

	// Copy data into GPU memory
	cudaMemcpy(atoms_d.x_list, atoms.x_list, PDH_acnt*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(atoms_d.y_list, atoms.y_list, PDH_acnt*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(atoms_d.z_list, atoms.z_list, PDH_acnt*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(histogram_d, histogram, num_buckets*sizeof(bucket), cudaMemcpyHostToDevice);

	const unsigned int numThreadsPerBlock = PDH_block_size;
	const unsigned int numBlocks = (PDH_acnt + numThreadsPerBlock - 1)/numThreadsPerBlock;

	// Query the maximum shared memory per block
	int device;
	cudaGetDevice(&device);
	int maxSharedMem;
	cudaDeviceGetAttribute(&maxSharedMem, cudaDevAttrMaxSharedMemoryPerBlock, device);

	// Calculate required shared memory and make sure it is within maximum shared memory
	const unsigned int sharedMemSize = (3 * numThreadsPerBlock * sizeof(double)) + (num_buckets * sizeof(int));
	if (sharedMemSize > maxSharedMem) {
		fprintf(stderr, "Error: Required shared memory (%u bytes) exceeds the maximum allowed (%d bytes).\n", sharedMemSize, maxSharedMem);
		exit(EXIT_FAILURE);
	}

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	SDH_kernel <<< numBlocks, numThreadsPerBlock, sharedMemSize >>> (atoms_d, histogram_d, PDH_acnt, PDH_res, num_buckets);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);
	printf("******** Total Running Time of Kernel = %0.5f ms ********\n", elapsedTime);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	// Copy data from GPU to host memory
	cudaMemcpy(histogram, histogram_d, num_buckets*sizeof(bucket), cudaMemcpyDeviceToHost);

	// Deallocate device (GPU) memory
	cudaFree(atoms_d.x_list);
	cudaFree(atoms_d.y_list);
	cudaFree(atoms_d.z_list);
	cudaFree(histogram_d);

	return 0;
}


/* 
	set a checkpoint and show the (natural) running time in seconds 
*/
double report_running_time(int CPU) {
	long sec_diff, usec_diff;
	gettimeofday(&endTime, &Idunno);
	sec_diff = endTime.tv_sec - startTime.tv_sec;
	usec_diff= endTime.tv_usec-startTime.tv_usec;
	if(usec_diff < 0) {
		sec_diff --;
		usec_diff += 1000000;
	}
	if (CPU)
		printf("Running time for CPU version: %ld.%06ld sec\n", sec_diff, usec_diff);
	else
		printf("Running time for GPU version: %ld.%06ld sec\n", sec_diff, usec_diff);
	return (double)(sec_diff*1.0 + usec_diff/1000000.0);
}


/* 
	print the counts in all buckets of the histogram 
*/
void output_histogram(){
	int i; 
	long long total_cnt = 0;
	for(i=0; i< num_buckets; i++) {
		if(i%5 == 0) /* we print 5 buckets in a row */
			printf("\n%02d: ", i);
		printf("%15lld ", histogram[i].d_cnt);
		total_cnt += histogram[i].d_cnt;
	  	/* we also want to make sure the total distance count is correct */
		if(i == num_buckets - 1)	
			printf("\n T:%lld \n", total_cnt);
		else printf("| ");
	}
    printf("\n");
}


/*
	Moves counts of each bucket in histogram to old_histogram and resets all buckets in histogram to 0
*/
int save_and_clear(bucket *old_histogram)
{
	for (int i = 0; i < num_buckets; i++)
	{
		old_histogram[i].d_cnt = histogram[i].d_cnt;    // Store old histogram value in old_histogram
		histogram[i].d_cnt = 0;                         // Reset histogram bucket to 0
	}

	return 0;
}

int get_diff_to_hist(bucket *hist)
{
	for (int i = 0; i < num_buckets; i++)
		histogram[i].d_cnt = abs(histogram[i].d_cnt - hist[i].d_cnt);   // Get distance between current histogram and histogram passed through hist parameter

	return 0;
}

int main(int argc, char **argv)
{
	int b, i;

	PDH_acnt = atoi(argv[1]);
	PDH_res	 = atof(argv[2]);
	PDH_block_size = atoi(argv[3]);

	num_buckets = (int)(BOX_SIZE * 1.732 / PDH_res) + 1;
	histogram = (bucket *)malloc(sizeof(bucket)*num_buckets);
	// Initialize histogram bucket counts
	for (b = 0; b < num_buckets; b++)
		histogram[b].d_cnt = 0;

    // Allocate memory for struct of lists
	atoms.x_list = (double *)malloc(sizeof(double)*PDH_acnt);
	atoms.y_list = (double *)malloc(sizeof(double)*PDH_acnt);
	atoms.z_list = (double *)malloc(sizeof(double)*PDH_acnt);

	srand(1);
	/* generate data following a uniform distribution */
	for(i = 0;  i < PDH_acnt; i++) {
		atoms.x_list[i] = ((double)(rand()) / RAND_MAX) * BOX_SIZE;
		atoms.y_list[i] = ((double)(rand()) / RAND_MAX) * BOX_SIZE;
		atoms.z_list[i] = ((double)(rand()) / RAND_MAX) * BOX_SIZE;
	}

    /* Compute histogram with CPU ONLY */
	gettimeofday(&startTime, &Idunno);  // start counting time
    PDH_baseline(); // call CPU single thread version to compute the histogram
	report_running_time(1);  // check the total running time (pass 1 to print for CPU)
    printf("Histogram for CPU Version:");
    output_histogram(); // print out the histogram

	/* Save bucket counts to new old_histogram array and reset histogram array*/
	bucket *old_histogram = (bucket *)malloc(sizeof(bucket)*num_buckets);
	save_and_clear(old_histogram);

    /* Compute histogram with GPU ONLY */
    PDH_parallel(); // call parallel GPU version to compute the histogram
    printf("Histogram for GPU Version:");
    output_histogram(); // print out the histogram
	
	/* get difference between CPU and GPU histograms */
    printf("Differences in Each Bucket Between CPU and GPU Histograms:");
	get_diff_to_hist(old_histogram);

	/* print out the difference histogram */
	output_histogram();
	
	/* Free Resources */
	free(histogram);
	free(old_histogram);
	free(atoms.x_list);
	free(atoms.y_list);
	free(atoms.z_list);
	
	return 0;
}
