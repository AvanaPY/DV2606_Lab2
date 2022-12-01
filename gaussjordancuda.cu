/***************************************************************************
 *
 * GPU version of Gauss-Jordan row reduction
 * Written by
 *  Emil Karlström, DVAMI19h
 *  Samuel Jonsson, DVAMI19h
 *
 ***************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <assert.h>

#define MAX_SIZE 4096
int MAX_BLOCK_SIZE;

typedef double matrix[MAX_SIZE][MAX_SIZE];

int	N;		/* matrix size		*/
int	maxnum;		/* max number of element*/
char* Init;		/* matrix init type	*/
int	PRINT;		/* print switch		*/
matrix	A;		/* matrix A		*/
double	b[MAX_SIZE];	/* vector b             */
double	y[MAX_SIZE];	/* vector y             */

/* Verifying that the computed matrix is correct */
int VERIFY;
matrix verify_A;
double verify_b[MAX_SIZE];
double verify_y[MAX_SIZE];

/* forward declarations */
void work(void);
void Init_Matrix(void);
void Print_Matrix(void);
void Init_Default(void);
void verify_result(void);
int Read_Options(int, char**);

int
main(int argc, char** argv)
{
    printf("Gauss Jordan GPU\n");
    clock_t timestart, timeend;

    Init_Default();		/* Init default values	*/
    Read_Options(argc, argv);	/* Read arguments	*/
    Init_Matrix();		/* Init the matrix	*/

    /* Prepare verification */
    memcpy(verify_A, A, sizeof(double) * MAX_SIZE * MAX_SIZE);
    memcpy(verify_b, b, sizeof(double) * MAX_SIZE);
    memcpy(verify_y, y, sizeof(double) * MAX_SIZE);

    timestart = clock();
    work();
    cudaDeviceSynchronize();
    timeend = clock();
    printf("Seconds used for computing: %f\n", (double)(timeend - timestart) / CLOCKS_PER_SEC);
    
    if (PRINT == 1)
        Print_Matrix();

    if(VERIFY == 1)
    {
        timestart = clock();
        verify_result();
        timeend = clock();
        printf("Seconds used for verification: %f\n", (double)(timeend - timestart) / CLOCKS_PER_SEC);
    }
}

__global__ void kernel_normalize_row(double* cuda_A, double* cuda_B, double* cuda_Y, int N, int k)
{
    int index = k + 1 + blockIdx.x * blockDim.x + threadIdx.x;
    if(index < N)
    {
        cuda_A[k * N + index] = cuda_A[k * N + index] / cuda_A[k * N + k];
    }
}

__global__ void kernel_norm_pivot(double* cuda_A, double* cuda_B, double* cuda_Y, int N, int k)
{
    cuda_Y[k] = cuda_B[k] / cuda_A[k * N + k];
    cuda_A[k * N + k] = 1;
}

__global__ void kernel_elimination(double* cuda_A, double* cuda_B, double* cuda_Y, int N, int k)
{
    int x = k + 1 + blockIdx.x * blockDim.x + threadIdx.x;
    int y = k + 1 + blockIdx.y * blockDim.y + threadIdx.y;

    // Boundary guard
    if((y < N) && (x < N))
        cuda_A[y * N + x] -= cuda_A[y * N + k] * cuda_A[k * N + x];
}

__global__ void kernel_eval_b(double* cuda_A, double* cuda_B, double* cuda_Y, int N, int k)
{
    int index = k + 1 + blockIdx.x * blockDim.x + threadIdx.x;
    if(index < N)
    {
        cuda_B[index] -= cuda_A[index * N + k] * cuda_Y[k];
        cuda_A[index * N + k] = 0.0;
    }
}
__global__ void kernel_gj_step(double* cuda_A, double* cuda_B, double* cuda_Y, int N, int k)
{
    int x = k + 1 + blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Boundary guard
    if((y < k) && (x < N))
        cuda_A[y * N + x] -= cuda_A[y * N + k] * cuda_A[k * N + x];
}
__global__ void kernel_gj_step2(double* cuda_A, double* cuda_B, double* cuda_Y, int N, int k)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index < k)
    {
        cuda_Y[index] -= cuda_A[index * N + k] * cuda_Y[k];
        cuda_A[index * N + k] = 0.0;
    }
}

void
work(void)
{
    /* Allocate and copy data to GPU */
    double *cuda_A, *cuda_B, *cuda_Y;
    cudaMalloc((void**)&cuda_A, sizeof(double) * N * N);
    cudaMalloc((void**)&cuda_B, sizeof(double) * N);
    cudaMalloc((void**)&cuda_Y, sizeof(double) * N);
    for(int k = 0; k < N; k++)
        cudaMemcpy(cuda_A + N * k, A[k], sizeof(double) * N, cudaMemcpyHostToDevice);
    
    cudaMemcpy(cuda_B, b, sizeof(double) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_Y, y, sizeof(double) * N, cudaMemcpyHostToDevice);

    /* GJ elimination */
    int block_size = MAX_BLOCK_SIZE * MAX_BLOCK_SIZE;
    int BLOCKS = max(1, N / block_size);
    
    dim3 blockDims(
        MAX_BLOCK_SIZE, 
        MAX_BLOCK_SIZE
    );
    dim3 gridDims(
        (int)ceil((float)N/(float)blockDims.x),
        (int)ceil((float)N/(float)blockDims.y)
    );

    clock_t start, end;

    start = clock();
    int k;
    for(k = 0; k < N; k++)
    {
        /* Normalize */
        kernel_normalize_row<<<BLOCKS, block_size>>>(cuda_A, cuda_B, cuda_Y, N, k);
        kernel_norm_pivot<<<1, 1>>>(cuda_A, cuda_B, cuda_Y, N, k);
        
        /* Standard elimination */
        kernel_elimination<<<gridDims, blockDims>>>(cuda_A, cuda_B, cuda_Y, N, k);
        kernel_eval_b<<<BLOCKS, block_size>>>(cuda_A, cuda_B, cuda_Y, N, k);

        /* Gauss Jordan step thingy and zeroing numbers before*/
        kernel_gj_step<<<gridDims, blockDims>>>(cuda_A, cuda_B, cuda_Y, N, k);
        kernel_gj_step2<<<BLOCKS, block_size>>>(cuda_A, cuda_B, cuda_Y, N, k);
    }
    end = clock();
    double difference = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Total raw computing time: %f\n", difference);

    /* Copy from GPU to RAM */

    for(int k = 0; k < N; k++)
        cudaMemcpy(A[k], cuda_A + N * k, sizeof(double) * N, cudaMemcpyDeviceToHost);
    cudaMemcpy(b, cuda_B, sizeof(double) * N, cudaMemcpyDeviceToHost);
    cudaMemcpy(y, cuda_Y, sizeof(double) * N, cudaMemcpyDeviceToHost);

    /* Print if we got any cool cuda errors */

    cudaError_t e = cudaGetLastError();
    const char* e_s = cudaGetErrorString(e);

    /* Free GPU memory; cuda is freeeeeeeeee~~~~~~~~ */
    cudaFree(cuda_A);
    cudaFree(cuda_B);
    cudaFree(cuda_Y);
}

void
Init_Matrix()
{
    int i, j;

    printf("\nsize       = %dx%d", N, N);
    printf("\nBlock size = <%d,%d>", MAX_BLOCK_SIZE, MAX_BLOCK_SIZE);
    printf("\nmaxnum     = %d", maxnum);
    printf("\nInit	   = %s", Init);
    printf("\nInitializing matrix...");

    if (strcmp(Init, "rand") == 0) {
        for (i = 0; i < N; i++) {
            for (j = 0; j < N; j++) {
                if (i == j) /* diagonal dominance */
                    A[i][j] = (double)(rand() % maxnum) + 5.0;
                else
                    A[i][j] = (double)(rand() % maxnum) + 1.0;
            }
        }
    }
    if (strcmp(Init, "fast") == 0) {
        for (i = 0; i < N; i++) {
            for (j = 0; j < N; j++) {
                if (i == j) /* diagonal dominance */
                    A[i][j] = 5.0;
                else
                    A[i][j] = 2.0;
            }
        }
    }

    /* Initialize vectors b and y */
    for (i = 0; i < N; i++) {
        b[i] = 2.0;
        y[i] = 1.0;
    }

    printf("done \n\n");
    if (PRINT == 1)
        Print_Matrix();
}

void
Print_Matrix()
{
    int i, j;

    printf("Matrix A:\n");
    for (i = 0; i < N; i++) {
        printf("[");
        for (j = 0; j < N; j++)
            printf(" %5.2f,", A[i][j]);
        printf("]\n");
    }
    
    printf("Vector b:\n[");
    for (j = 0; j < N; j++)
        printf(" %5.2f,", b[j]);
    printf("]\n");

    printf("Vector y:\n[");
    for (j = 0; j < N; j++)
        printf(" %5.2f,", y[j]);
    printf("]\n");
    printf("\n\n");
}

void
Init_Default()
{
    N = 2048;
    Init = "fast";
    maxnum = 15.0;
    PRINT = 0;
    VERIFY = 0;

    MAX_BLOCK_SIZE = 32;
}

int
Read_Options(int argc, char** argv)
{
    char* prog;

    prog = *argv;
    while (++argv, --argc > 0)
        if (**argv == '-')
            switch (*++ * argv) {
            case 'n':
                --argc;
                N = atoi(*++argv);
                break;
            case 'h':
                printf("\nHELP: try sor -u \n\n");
                exit(0);
                break;
            case 'u':
                printf("\nUsage: gaussian [-n problemsize]\n");
                printf("           [-D] show default values \n");
                printf("           [-h] help \n");
                printf("           [-I init_type] fast/rand \n");
                printf("           [-m maxnum] max random no \n");
                printf("           [-P print_switch] 0/1 \n");
                exit(0);
                break;
            case 'D':
                printf("\nDefault:  n         = %d ", N);
                printf("\n          Init      = rand");
                printf("\n          maxnum    = 5 ");
                printf("\n          P         = 0 \n\n");
                exit(0);
                break;
            case 'I':
                --argc;
                Init = *++argv;
                break;
            case 'm':
                --argc;
                maxnum = atoi(*++argv);
                break;
            case 'P':
                --argc;
                PRINT = atoi(*++argv);
                break;
            case 't':
                --argc;
                MAX_BLOCK_SIZE = atoi(*++argv);
                break;
            case 'v':
                --argc;
                VERIFY = atoi(*++argv);
                break;
            default:
                printf("%s: ignored option: -%s\n", prog, *argv);
                printf("HELP: try %s -u \n\n", prog);
                break;
            }
    return 0;
}

double _round_to_decimals(double value, int decimals)
{
    int fac = pow(10, decimals - 1);
    return round((value * decimals) / decimals);
}

void verify_result()
{
    printf("Verifying result...\n");
    /* Gaussian elimination algorithm, Algo 8.4 from Grama */
    int k, j, i;
    for (k = 0; k < N; k++) { /* Outer loop */
        for (j = k + 1; j < N; j++)
            verify_A[k][j] = verify_A[k][j] / verify_A[k][k]; /* Division step */
        verify_y[k] = verify_b[k] / verify_A[k][k];
        verify_A[k][k] = 1.0;
        for (i = k + 1; i < N; i++) {
            for (j = k + 1; j < N; j++)
                verify_A[i][j] = verify_A[i][j] - verify_A[i][k] * verify_A[k][j]; /* Elimination step */
            verify_b[i] = verify_b[i] - verify_A[i][k] * verify_y[k];
            verify_A[i][k] = 0.0;
        }
        for (i = 0; i < k; i++) {
            for (j = k + 1; j < N; j++)
                verify_A[i][j] = verify_A[i][j] - verify_A[i][k] * verify_A[k][j]; /* Additional Elimination for Gauss-Jordan */
            verify_y[i] = verify_y[i] - verify_A[i][k] * verify_y[k];
            verify_A[i][k] = 0.0;
        }
    }
    printf("\tComputed correct matrix.\n");

    /* Print original matrix */

    if(PRINT == 1)
    {
        printf("Matrix A:\n");
        for (i = 0; i < N; i++) {
            printf("[");
            for (j = 0; j < N; j++)
                printf(" %5.15f,", A[i][j]);
            printf("]\n");
        }

        printf("\n");
        for (i = 0; i < N; i++) {
            printf("[");
            for (j = 0; j < N; j++)
                printf(" %5.15f,", verify_A[i][j]);
            printf("]\n");
        }
        
        printf("bs:\n[");
        for (j = 0; j < N; j++)
            printf(" %5.15f,", b[j]);
        printf("]\n[");
        for (j = 0; j < N; j++)
            printf(" %5.15f,", verify_b[j]);
        printf("]\n");

        printf("ys:\n[");
        for (j = 0; j < N; j++)
            printf(" %5.15f,", y[j]);
        printf("]\n[");
        for (j = 0; j < N; j++)
            printf(" %5.15f,", verify_y[j]);
        printf("]\n");
        printf("\n\n");
    }

    /* Print verify matrix */

    /* Verify they are still the same */
    int decimals = 10;
    for(int i = 0; i < N; i++)
    {
        for(int j = 0; j < N; j++)
            assert(_round_to_decimals(verify_A[i][j], decimals) == _round_to_decimals(A[i][j], decimals) && "verify_A not equal to A");
        assert(_round_to_decimals(verify_b[i], decimals) == _round_to_decimals(b[i], decimals) && "verify_b not equal to b");
        assert(_round_to_decimals(verify_y[i], decimals) == _round_to_decimals(y[i], decimals) && "verify_y not equal to y");
    }

    printf("\tPassed verification to %d decimals\n", decimals);
}