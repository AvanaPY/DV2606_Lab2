# DV2606 - Lab2 GPU programming

Written by:
* Emil Karlström, DVAMI19h
* Samuel Jonsson, DVAMI19h


# Make

`make compile_cuda` to compile all CUDA programs.

`make clean` to clean the directory of any executables.

`make zip` to zip all the relevant files into a zip file.

`make unzip` to unzip the zip file generated from `make zip`. Mainly used to verify that the files in the zip file are the correct ones.

### Optionals

To compile all C/C++ programs run `make compile_c`. Please note that this requires you to copy the original gauss-jordan C implementation and the odd-even-sort C++ implementation into the directory and to rename them to `gaussjordanseq.c` and `oddevensort.cpp` respectively. Used by us to easily compile the code given to us in the assignment, hence it is not directly part of the assignment turn-in. 

# Files

`gaussjordancuda.a` is the CUDA Gauss-Jordan implementation. See below for new flags you can use when running the program.

`oddevensortcu.a` is the CUDA odd-even-sort implementation that uses multi-kernel calls.

`oddevensort_one_block_block.a` is the CUDA odd-even-sort implementation that uses a single kernel call and divides the array into chunks and maps each thread to a chunk.

`oddevensort_one_block_stride.a` is the CUDA odd-even-sort implementation that uses a single kernel call and divides using strides rather than chunks to map threads to pairs.

# Featured new flags in Gauss-Jordan CUDA

* `-t` controlls the amount of threads to use for each dimension of a 2D block of threads. The total amount of threads per block is the square of this number, i.e `-t 4` would use a 4x4 blocks, i.e 16 threads per block.

* `-v` controlls whether or not to perform verification of the computed result. This uses the gauss-jordan sequential algorithm from Grama (Algorithm 8.4) to compute the correct result, then compares the results to 10 decimal places. The reason we choose `10` decimals is that we consider that being enough accuracy whilst still having some spare room in case of floating point precision rounding errors were to occur. 

    * `-v 1` means to compute verification and `-v 0` is not to. 
    * The verification only compares the resulting `y` vector. 