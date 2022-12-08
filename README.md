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

By default this uses a matrix of 2048x2048 and block dimensions of 32x32, i.e 1024 threads per block.

`oddevensortcu.a` is the CUDA odd-even-sort implementation that uses multi-kernel calls. 

By default this uses a list of 2^19 items.

`oddevensort_one_block_block.a` is the CUDA odd-even-sort implementation that uses a single kernel call and divides the array into chunks and maps each thread to a chunk. 

By default this uses a list of 100 000 items as 2^19 items takes too long to compute.

`oddevensort_one_block_stride.a` is the CUDA odd-even-sort implementation that uses a single kernel call and divides using strides rather than chunks to map threads to pairs. 

By default this uses a list of 100 000 items as 2^19 items takes a long time compute.

# New features in Gauss-Jordan CUDA

## Flags 

* `-t` controlls the amount of threads to use for each dimension of a 2D block of threads. The total amount of threads per block is the square of this number, i.e `-t 4` would use a 4x4 blocks, i.e 16 threads per block. Defaults to `32`.

* `-v` controlls whether or not to perform verification of the computed result. This uses the gauss-jordan sequential algorithm from Grama (Algorithm 8.4) to compute the correct result, then compares the results to 10 decimal places. The reason we choose `10` decimals is that we consider that being enough accuracy whilst still having some spare room in case of floating point precision rounding errors were to occur. Defaults to `0`.

    * `-v 1` means to compute verification and `-v 0` is not to. 
    * The verification only compares the resulting `y` vector. 

## Macros

* `gaussjordancuda.cu` now contains a macro named `__PROFILE__z` by default. By renaming it to `__PROFILE__` (without the `z`) one can profile the different stages of the program using the `chrono` library.