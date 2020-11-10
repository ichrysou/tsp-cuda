/* TSP Solution with CUDA
 * Copyright (c) 2011 Elias Kyrlies-Chrysoulidis
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#define N 13
#define M N *(N - 1) / 2
#define F factorialRec(N)
#define LEFT 1
#define RIGHT 0
#define S 10
#define NoT factorialRec(S)
#define pi 3.1415926536

typedef struct {
    float x;
    float y;
    int index;
    int decoy;
} city;

typedef struct {
    int items_ind[N];  //
    int itemsi[N];
    int tail;

} queue;

int D_MapHost(int a, int b);
float RouteCalcHost(city *cities, float *edges);
long long factorialRec(int n);
void DistInit(city *cities, float *dist);
float DistCalc(city c1, city c2);
void CitiesInit(city *cities);
__device__ int D_Map(int a, int b);
__device__ float RouteCalc(city *cities, float *edges);
__global__ void TSP_Kernel(city *devPtrCit, city *devPtrCitRes, float *devPtrEdge, float *devPtrRes, long long *devPtrFac, int *res,
                           int thr_num);
float Test(city *cities, float *edges);

int main(int argc, char *argv[]) {
    struct timeval time1, time2;
    float dt;
    int i;

    // Initialize cities map. Random 2-D points
    city *cities = (city *)malloc(N * sizeof(city));
    float *edges = (float *)malloc(M * sizeof(float));
    float test;

    if (argc == 2) {
        if (!(strcmp(argv[1], "test"))) test = Test(cities, edges);
    } else {
        CitiesInit(cities);
        DistInit(cities, edges);
    }

    // Initialize factorials LUT
    long long factorials[N];

    //============Memory Init============
    //------ Allocate device memroy------
    //===================================

    // Allocate memory for cities
    city *devPtrCit;
    cudaMalloc((void **)&devPtrCit, (size_t)N * sizeof(city));
    // Copy cities to device
    cudaMemcpy(devPtrCit, cities, (size_t)N * sizeof(city), cudaMemcpyHostToDevice);
    // Allocate memory for result Route
    city *devPtrCitRes;
    cudaMalloc((void **)&devPtrCitRes, (size_t)N * sizeof(city));
    // Allocate memory for edges
    float *devPtrEdge;
    cudaMalloc(&devPtrEdge, (size_t)M * sizeof(float));
    // Copy edges to device
    cudaMemcpy(devPtrEdge, edges, (size_t)M * sizeof(float), cudaMemcpyHostToDevice);
    // Allocate memory for results
    float *devPtrRes;
    cudaMalloc(&devPtrRes, (size_t)sizeof(float));
    // Allocate memory for factorials
    long long *devPtrFac;
    cudaMalloc(&devPtrFac, (size_t)N * sizeof(long long));
    // Allocate memory for debug
    int *res;
    cudaMalloc(&res, (size_t)sizeof(int));

    // Let the host calculate the factorials
    for (i = 0; i < N; i++) {
        factorials[i] = factorialRec(i + 1);
        printf("%lld ", factorials[i]);
    }

    cudaMemcpy(devPtrFac, factorials, (size_t)N * sizeof(long long), cudaMemcpyHostToDevice);

    // Initialize the kernel
    // Calculate number of threads to be used
    int grx, gry;
    grx = (int)(sqrt(NoT) / 16 + 1);
    gry = (int)(sqrt(NoT) / 16 + 1);
    printf("grx = %d , gry = %d , NoT = %lld\n", grx, gry, NoT);
    dim3 dimBlock(16, 16, 1);
    dim3 dimGrid(grx, gry, 1);

    // Start the kernel
    cudaThreadSynchronize();
    gettimeofday(&time1, NULL);
    TSP_Kernel<<<dimGrid, dimBlock>>>(devPtrCit, devPtrCitRes, devPtrEdge, devPtrRes, devPtrFac, res, NoT);
    cudaThreadSynchronize();
    gettimeofday(&time2, NULL);

    dt = (float)(time2.tv_sec - time1.tv_sec + (time2.tv_usec - time1.tv_usec) / 1.0e6);
    printf("Cuda kernel execution time: %f\n", dt);

    // Copy results back to host
    float *resHost = (float *)malloc((size_t)sizeof(float));
    int *resHostInt = (int *)malloc((size_t)sizeof(int));
    cudaMemcpy(resHost, devPtrRes, (size_t)sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(resHostInt, res, (size_t)sizeof(int), cudaMemcpyDeviceToHost);

    city *mincities = (city *)malloc((size_t)N * sizeof(city));
    cudaMemcpy(mincities, devPtrCitRes, (size_t)N * sizeof(city), cudaMemcpyDeviceToHost);

    // Print results
    printf("Direction Pointer: %d\n", (*resHostInt));

    for (i = 0; i < N; i++) {
        //    printf("__%f__ ",resHost[i]);
        printf("__%d__", mincities[i].index);
        //  printf("__%d__",resHostInt[i]);
    }
    printf("Kernel calculated result: %f\n", (*resHost));
    printf("Recaclulate result: %f\n", RouteCalcHost(mincities, edges));
    if (argc == 2) {
        if (int(test * 100) == int((*resHost) * 100))
            printf("Test PASSED!\n");
        else
            printf("Test FAILED :(\n");
    }

    cudaFree(devPtrCit);
    cudaFree(devPtrEdge);
    cudaFree(devPtrRes);
    cudaFree(devPtrFac);
    cudaFree(res);
    //  free(resHost);
}

// The kernel works well for NoT threads where NoT = S! and 0<S<N
__global__ void TSP_Kernel(city *devPtrCit, city *devPtrCitRes, float *devPtrEdge, float *devPtrRes, long long *devPtrFac, int *res,
                           int thr_num) {
    //============================================================================
    //----------------------------Declerations------------------------------------
    //============================================================================

    // blockDim.x = 16
    // blockDim.y = 16
    int thread_x = blockDim.x * blockIdx.x + threadIdx.x;
    int thread_y = blockDim.y * blockIdx.y + threadIdx.y;

    int tid = blockDim.y * gridDim.y * thread_x + thread_y;

    __shared__ long long factorials[N];
    __shared__ float edges[M];
    __shared__ float Res[16 * 16];

    city cities[N];      // local array. Holds current permutation
    city mincit[N];      //  >>  . Holds optimal permutation so far
    int fac_rep[N - 1];  //  >>  .Holds factorial representation for each thread
    long dirpoint;       // Holds direction pointers. Bit set for LEFT.
    int temp, cntr, min, mini;
    int i, j;
    city swp;
    bool flag;
    float dist;
    queue movable;  // queue holds all movable items

    //============================================================================
    //----------------------Initialize Krenel-------------------------------------
    //============================================================================

    // store initial sequence from global to local memory
    for (int i = 0; i < N; i++) cities[i] = devPtrCit[i];

    // store factorials and edges to shared memory
    __syncthreads();
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        for (i = 0; i < M; i++) {
            if (i < N) factorials[i] = devPtrFac[i];
            edges[i] = devPtrEdge[i];
        }
    }
    __syncthreads();

    // Every thread calculates tid's factorial representation
    // thus producing a unique sequence
    temp = tid;
    for (i = N - 2; i >= 0; i--) {
        fac_rep[i] = temp / factorials[i];
        temp = temp % factorials[i];
    }

    // Get initial permutation for each thread
    for (i = 0; i < N - 1; i++) {
        cntr = i;
        while (fac_rep[i] != 0) {
            swp = cities[cntr];
            cities[cntr] = cities[cntr + 1];
            cities[cntr + 1] = swp;
            cntr--;
            fac_rep[i]--;
        }
    }
    __syncthreads();

    // Calculate total distance for each thread
    // Store in shared memory
    if (tid < thr_num) {
        dist = RouteCalc(cities, edges);
        Res[threadIdx.x * blockDim.y + threadIdx.y] = dist;
        for (i = 0; i < N; i++) mincit[i] = cities[i];
    }
    __syncthreads();

    //---Initialization Done....------------
    // Main body.....

    //===========================================================================
    //-------------- Permute till no movable elements exist ----------------
    //===========================================================================

    // Initialize direction pointer
    dirpoint = 0xffffffff;  //<--suitable for up to 32 cities
    cntr = 0;               //<---obsolete

    //--------------- Iterate till no movable items exist------------------------
    while (1) {
        // 1. Find movable elements
        movable.tail = 0;
        for (i = 0; i < N; i++) {
            if (cities[i].index > S - 1) {
                if ((dirpoint >> (cities[i].index)) & 1 == LEFT) {
                    for (j = i; j >= 0; j--) {
                        if (cities[j].index < cities[i].index) {
                            movable.items_ind[movable.tail] = cities[i].index;
                            movable.itemsi[movable.tail] = i;
                            movable.tail++;
                            break;
                        }
                    }
                } else {
                    for (j = i; j < N; j++) {
                        if (cities[j].index < cities[i].index) {
                            movable.items_ind[movable.tail] = cities[i].index;
                            movable.itemsi[movable.tail] = i;
                            movable.tail++;
                            break;
                        }
                    }
                }
            }
        }
        // 2. If no movable items found break loop..Job is done!
        if (movable.tail == 0) break;

        // 3. Find minimum movable item
        min = movable.items_ind[0];
        mini = movable.itemsi[0];
        for (i = 1; i < movable.tail; i++) {
            if (movable.items_ind[i] < min) {
                min = movable.items_ind[i];
                mini = movable.itemsi[i];
            }
        }
        // 4. Interchange smallest movable element
        if ((dirpoint >> min) & 1 == LEFT) {
            for (j = mini; j >= 0; j--) {
                if (cities[j].index < min) {
                    swp = cities[mini];
                    cities[mini] = cities[j];
                    cities[j] = swp;
                    mini = j;
                    break;
                }
            }

        } else {
            for (j = mini; j < N; j++) {
                if (cities[j].index < min) {
                    swp = cities[mini];
                    cities[mini] = cities[j];
                    cities[j] = swp;
                    mini = j;
                    break;
                }
            }
        }

        // 5. Toggle dirpointer
        for (i = 0; i < N; i++) {
            if (cities[i].index > S - 1 && cities[i].index < min) {
                dirpoint ^= (1 << cities[i].index);
            }
        }

        // 6. Renew result tables
        if (tid < thr_num) {
            dist = RouteCalc(cities, edges);

            if (dist < Res[threadIdx.x * blockDim.y + threadIdx.y]) {
                Res[threadIdx.x * blockDim.y + threadIdx.y] = dist;  // RouteCalc(cities,edges);//dist;
                for (i = 0; i < N; i++) mincit[i] = cities[i];
            }
        }
    }
    //============================================================================
    //-----------------------while loop ends here---------------------------------
    //============================================================================
    // Buy now every thread has stored optimal sequence in mincities[i].Minimum
    // distance encountered is in Res[localID].
    // All you 've got to do is find the min of the mins globaly.

    // Store distance as integer. Necessary in order to use CUDA atomic operations.
    if (tid == 0) {
        (*res) = int(1000 * dist);  // dist;//dirpoint;
        for (i = 0; i < N; i++) devPtrCitRes[i] = mincit[i];
    }
    __syncthreads();

    // Find the min in each block
    flag = true;
    if (tid < thr_num) {
        for (i = 0; i < 16 * 16; i++) {
            if (Res[threadIdx.x * blockDim.y + threadIdx.y] > Res[i]) {
                flag = false;
                break;
            }
        }
    }
    // Buy the end of this code block (*res) holds optimal thread's tid.
    if (tid < thr_num) {
        if (flag == true) {
            atomicMin(res, (int)(1000 * Res[threadIdx.x * blockDim.y + threadIdx.y]));
            atomicCAS(res, (int)(1000 * Res[threadIdx.x * blockDim.y + threadIdx.y]), tid);
            if (tid == (*res)) {
                for (i = 0; i < N; i++) devPtrCitRes[i] = mincit[i];
                (*devPtrRes) = Res[threadIdx.x * blockDim.y + threadIdx.y];
            }
        }
    }

    // Done finding min..
    // Kernel exits...
}

void DistInit(city *cities, float *edges) {
    int i, j;
    int index = 0;

    for (j = 0; j < N; j++) {
        for (i = j + 1; i < N; i++) {
            edges[index] = DistCalc(cities[j], cities[i]);
            index++;
        }
    }
}

//==========================================================

float DistCalc(city c1, city c2) { return (sqrt(pow((double)(c1.x - c2.x), 2) + pow((double)(c1.y - c2.y), 2))); }

//==========================================================

void CitiesInit(city *cities) {
    int i;

    srand(time(NULL));

    for (i = 0; i < N; i++) {
        cities[i].x = (rand() % 100);
        cities[i].y = (rand() % 100);
        cities[i].index = i;
    }
}

//==========================================================

__device__ int D_Map(int a, int b) {
    int i;
    int ind = 0;
    if (a < b) {
        for (i = 0; i < a; i++) ind += (N - 1 - i);
    } else {
        for (i = 0; i < b; i++) ind += (N - 1 - i);
    }
    ind += abs(b - a);
    ind--;
    return (ind);
}

//++++++++++++++++++++++++++++++++++++++++++++++++++

int D_MapHost(int a, int b) {
    int i;
    int ind = 0;
    if (a < b) {
        for (i = 0; i < a; i++) ind += (N - 1 - i);
    } else {
        for (i = 0; i < b; i++) ind += (N - 1 - i);
    }
    ind += abs(b - a);
    ind--;
    return (ind);
}

//==========================================================

// Calculates total distance
__device__ float RouteCalc(city *cities, float *edges) {
    int i;
    float dist = 0;
    for (i = 0; i < N - 1; i++) dist += edges[D_Map(cities[i].index, cities[i + 1].index)];
    return (dist);
}

//++++++++++++++++++++++++++++++++++++++++++++++++++

float RouteCalcHost(city *cities, float *edges) {
    int i;
    float dist = 0;
    for (i = 0; i < N - 1; i++) dist += edges[D_MapHost(cities[i].index, cities[i + 1].index)];
    return (dist);
}

//==========================================================

long long factorialRec(int n) {
    if (n <= 1)
        return 1;
    else
        return n * factorialRec(n - 1);
}

//=========================================================

float Test(city *cities, float *edges) {
    int i;
    for (i = 0; i < N; i++) {
        cities[i].x = cos((2 * pi / N) * i);
        cities[i].y = sin((2 * pi / N) * i);
        cities[i].index = i;
    }
    DistInit(cities, edges);
    printf("Minimum route length: %f\n", RouteCalcHost(cities, edges));
    return (RouteCalcHost(cities, edges));
}
