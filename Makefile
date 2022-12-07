GCC=gcc
CFLAGS=-O2

CUDA=/usr/local/cuda-11.8/bin/nvcc
CUDAUSRBIN=/usr/bin/nvcc

compile_all: compile_cpp compile_cuda

compile_cpp:
	@$(GCC) $(CFLAGS) -o gaussjordancpp.a gaussjordanseq.c
	@g++ $(CFLAGS) -o oddevensortcpp.a oddevensort.cpp
	
compile_cuda:
	@$(CUDAUSRBIN) -o gaussjordancuda.a gaussjordancuda.cu
	@$(CUDAUSRBIN) -o oddevensortcu.a oddevensort.cu
	@$(CUDAUSRBIN) -o oddevensortcuda_one_block_block.a oddevensortcuda_one_block_block.cu
	@$(CUDAUSRBIN) -o oddevensortcuda_one_block_stride.a oddevensortcuda_one_block_stride.cu

%.o: %.c
	$(GCC) $(CFLAGS) $< -c -o $@

clean:
	@rm -rf *.a