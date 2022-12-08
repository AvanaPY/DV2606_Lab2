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
#define THREADS 1024

void print_array_status(int* v, int vsize);
void print_sort_status(int* v, int vsize);
void init_array(int* v, int vsize);
void odd_even_sort(int* v, int vsize);

/*
    This is our kernel function, it computes the odd-even sort
    and syncs all GPU threads using __syncthreads() every iteration of i
*/
__global__ void vector_odd_even_sort(int* cuda_v, int vsize, int num_threads)
{
    int j, tmp;
    int pairs_per_block = (int)ceil((float)vsize / num_threads);
    int thread_offset = threadIdx.x * 2 * pairs_per_block;
    for(int i = 1; i <= vsize; i++)
    {
        /*  Compute which pair this thread shall start sorting
            and at the end of the loop we explicitly increment j by 2
            so it starts working on the next pair in the chunk */
        j = i % 2 + thread_offset;
        for(int k = 0; k < pairs_per_block; k++) {
            /* Boundary guard */
            if(j > vsize - 2)
                break;

            /* Perform comparison and swap */
            if(cuda_v[j] > cuda_v[j + 1])
            {
                tmp = cuda_v[j];
                cuda_v[j] = cuda_v[j + 1];
                cuda_v[j+1] = tmp;
            }
            /* Update J very uglyly */
            j += 2;
        }
        /* Sync the threads in this block */
        __syncthreads();
    }
}

int main(void)
{
    printf("Be warned that this program takes about 3-9.5x longer to complete than the stride-using version. Memory sucks.\n");
    constexpr unsigned int size = 100000;
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
    /* My computer only supports up to 1024 threads per block, 
        you can chance if you want but no guarantees your computer
        won't explode */

    vector_odd_even_sort<<<1, THREADS>>>(cuda_v, vsize, THREADS);
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