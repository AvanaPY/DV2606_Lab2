GCC=gcc
CFLAGS=-O2

CUDA=/usr/local/cuda-11.8/bin/nvcc
CUDAUSRBIN=/usr/bin/nvcc
	
compile_cuda:
	@$(CUDA) -o gaussjordancuda.a gaussjordancuda.cu
	@$(CUDA) -o oddevensortcu.a oddevensort.cu
	@$(CUDA) -o oddevensortcuda_one_block_block.a oddevensortcuda_one_block_block.cu
	@$(CUDA) -o oddevensortcuda_one_block_stride.a oddevensortcuda_one_block_stride.cu

compile_c:
	@$(GCC) $(CFLAGS) -o gaussjordancpp.a gaussjordanseq.c
	@g++ $(CFLAGS) -o oddevensortcpp.a oddevensort.cpp

%.o: %.c
	$(GCC) $(CFLAGS) $< -c -o $@

clean:
	@rm -rf *.a
	@rm -rf assignment assignment.zip

zip:
	@rm -rf assignment
	@mkdir assignment
	@cp *.cu *.pdf Makefile README.md assignment
	@zip assignment.zip assignment/*
	@rm -rf assignment

unzip: zip
	@unzip assignment.zip
