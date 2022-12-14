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

#define MAXNUM 65536 
#define THREADS_PER_BLOCK 512
#define BLOCKS 32

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
    int thread_offset = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    int stride_offset = BLOCKS * THREADS_PER_BLOCK * 2;
    /* Compute which pairs this thread shall sort */
    for(j = i % 2 + thread_offset; j < vsize - 1; j += stride_offset){
        if(cuda_v[j] > cuda_v[j + 1])
        {
            tmp = cuda_v[j];
            cuda_v[j] = cuda_v[j + 1];
            cuda_v[j+1] = tmp;
        }
    }
}

int main(void)
{
    constexpr unsigned int size = 2<<19;
    int v[size];
    init_array(v, size);

    odd_even_sort(v, size);
    print_sort_status(v, size);
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

    /* 
        This is our outer loop in the odd-even sorting algorithm
        It synchronises the GPU blocks which are distributed 
        to compute the sorting of one iteration of the odd-even sorting algorithm
    */
    for(int i = 1; i <= vsize; i++)
        vector_odd_even_sort<<<BLOCKS, THREADS_PER_BLOCK>>>(cuda_v, vsize, i);
    cudaDeviceSynchronize();
    
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