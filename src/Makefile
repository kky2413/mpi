NVCC = nvcc
MPICC = /usr/bin/mpicc.openmpi
MPI_INC_PATH = -I/usr/lib/x86_64-linux-gnu/openmpi/include/
MPI_LIB_PATH = -L/usr/lib/x86_64-linux-gnu/openmpi/lib
CUDA_INC_PATH = -I/usr/local/cuda/include
CUDA_LIB_PATH = -L/usr/local/cuda/lib64
LINK_OPS = -c -w
DEBUG = -g -O3
CUDA_FLAGS = --fmad=false --prec-div=true --prec-sqrt=true
CLIBS = -lm -lstdc++
CUDA_LIBS = -lcudart
OBJECTS = PFM_MPI.o PFM_MPI_GPU.o
MPI_CUDA = pfm_mpi_cuda.out 

# 최종 링크 시 CUDA_FLAGS 제거 (gcc/mpicc는 CUDA_FLAGS를 모르기 때문)
$(MPI_CUDA): $(OBJECTS)
	$(MPICC) $(DEBUG) $(OBJECTS) $(CUDA_LIB_PATH) $(CUDA_LIBS) $(CLIBS) -o $(MPI_CUDA)

PFM_MPI.o: PFM_MPI.c Initialize_domain.cu savetimestep.cu
	$(MPICC) $(DEBUG) $(LINK_OPS) PFM_MPI.c -o PFM_MPI.o $(CUDA_INC_PATH)

# CUDA_FLAGS는 nvcc 호출할 때만 추가해주기
PFM_MPI_GPU.o: PFM_MPI_GPU.cu savetimestep.cu kernels_cuda.cu
	$(NVCC) $(DEBUG) $(CUDA_FLAGS) $(LINK_OPS) PFM_MPI_GPU.cu -o PFM_MPI_GPU.o $(MPI_INC_PATH)

clean:
	rm -f $(MPI_CUDA) $(OBJECTS)

remove:
	rm -rf *.dat nohup.out

remove_images:
	rm -rf *.png

