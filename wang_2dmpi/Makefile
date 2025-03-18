NVCC = nvcc
MPICC = /usr/bin/mpicc.openmpi
MPI_INC_PATH = -I/usr/lib/x86_64-linux-gnu/openmpi/include/
MPI_LIB_PATH = -L/usr/lib/x86_64-linux-gnu/openmpi/lib
CUDA_INC_PATH = -I/usr/local/cuda/include
CUDA_LIB_PATH = -L/usr/local/cuda/lib64
LINK_OPS = -c -w
DEBUG = -g
CLIBS=-lm -lstdc++
CUDA_LIBS = -lcudart
OBJECTS = PFM_MPI.o PFM_MPI_GPU.o
MPI_CUDA = pfm_mpi_cuda.out 
MPI_CUDA: $(OBJECTS)
	$(MPICC) $(DEBUG) $(OBJECTS) $(CUDA_LIB_PATH) $(CUDA_LIBS) $(CLIBS) -o $(MPI_CUDA)

PFM_MPI.o: PFM_MPI.c Initialize_domain.cu savetimestep.cu
	$(MPICC) $(DEBUG) $(LINK_OPS) PFM_MPI.c -o PFM_MPI.o $(CUDA_INC_PATH)

PFM_MPI_GPU.o: PFM_MPI_GPU.cu savetimestep.cu kernels_cuda.cu
	$(NVCC) $(DEBUG) $(LINK_OPS) PFM_MPI_GPU.cu -o PFM_MPI_GPU.o $(MPI_INC_PATH)

clean:
	rm $(MPI_CUDA) $(OBJECTS)
 
remove:
	rm -rf *.dat nohup.out
# To remove the generated images (after post processing)
remove_images:
	rm -rf *.png
