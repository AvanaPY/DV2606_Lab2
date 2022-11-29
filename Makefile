GCC=gcc
CFLAGS=-O2

CUDA=/usr/local/cuda-11.8/bin/nvcc

compile_cpp:
	@$(GCC) $(CFLAGS) -o gaussjordancpp.a gaussjordanseq.c
	@g++ $(CFLAGS) -o oddevensortcpp.a oddevensort.cpp
	
compile_cuda:
	@$(CUDA) -o gaussjordancuda.a gaussjordancuda.cu
	@$(CUDA) -o oddevensortcu.a oddevensort.cu
compile_all: compile_cpp compile_cuda

%.o: %.c
	$(GCC) $(CFLAGS) $< -c -o $@

clean:
	@rm -rf *.a