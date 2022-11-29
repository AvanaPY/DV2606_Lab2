/***************************************************************************
 *
 * GPU version of Odd Even Sorting
 * Written by
 *  Emil Karlström, DVAMI19h
 *  Samuel Jonsson, DVAMI19h
 *
 ***************************************************************************/

#include <iostream>
#include <chrono>
#include <math.h>

#define MAXNUM 65564
#define MAX_THREADS_PER_BLOCK 1024

void print_array_status(int* v, int vsize);
void print_sort_status(int* v, int vsize);
void init_array(int* v, int vsize);
void odd_even_sort(int* v, int vsize);

/*
    This is our kernel function, it computes one iteration
    of the odd-even sorting algorithm
*/
__global__ void vector_odd_even_sort(int* cuda_v, int vsize, int i)
{
    int j, tmp;

    /* Compute which pair this thread shall sort */
    j = i % 2 + (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    if(j < vsize - 1)
    {
        if(cuda_v[j] > cuda_v[j + 1])
        {
            tmp = cuda_v[j];
            cuda_v[j] = cuda_v[j + 1];
            cuda_v[j+1] = tmp;
        }
    }
    /* Synch the threads in this block */
    __syncthreads();
}

int main(void)
{
    constexpr unsigned int size = 2<<19;
    int v[size];
    init_array(v, size);

    odd_even_sort(v, size);
    
    print_sort_status(v, size);
    cudaDeviceSynchronize();
    return 0;
}

/*
    Main API function
*/
void odd_even_sort(int* v, int vsize)
{
    // Step 1: Copy to GPU unit

    int* cuda_v;
    cudaMalloc((void**)&cuda_v, sizeof(int) * vsize);
    cudaMemcpy(cuda_v, v, sizeof(int) * vsize, cudaMemcpyHostToDevice);

    // Sort

    /* Compute how many blocks we have to use */
    int BLOCKS = max(1, vsize / MAX_THREADS_PER_BLOCK);

    /* 
        This is our outer loop in the odd-even sorting algorithm
        It synchronises the GPU blocks which are distributed 
        to compute the sorting of one iteration of the odd-even sorting algorithm
    */
    for(int i = 1; i <= vsize; i++)
        vector_odd_even_sort<<<BLOCKS, MAX_THREADS_PER_BLOCK>>>(cuda_v, vsize, i);

    // Copy back to host memory
    cudaMemcpy(v, cuda_v, sizeof(int) * vsize, cudaMemcpyDeviceToHost);

    // Free memory
    cudaFree(cuda_v);
}

/* Print the array values */
void print_array_status(int* v, int vsize)
{
    for(int i = 0; i < vsize; i++)
        printf("%d,", v[i]);
    printf("\n");
}

/* Print the array's sort status */
void print_sort_status(int* v, int vsize)
{
    int sorted = 1;
    for(int i = 0; i < vsize - 1; i++)
        sorted &= v[i] <= v[i + 1];
    std::cout << "The input is sorted?: " << (sorted == 1 ? "True" : "False") << std::endl;
}

/* (Psuedo)-randomly initialise the array */
void init_array(int* v, int vsize)
{
    for(int i = 0; i < vsize; i++)
        v[i] = (rand() % MAXNUM);
}