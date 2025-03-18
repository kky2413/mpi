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
OBJECTS = main.o main_gpu.o
MPI_CUDA = spino.out 
MPI_CUDA: $(OBJECTS)
	$(MPICC) $(DEBUG) $(OBJECTS) $(CUDA_LIB_PATH) $(CUDA_LIBS) $(CLIBS) -o $(MPI_CUDA)

main.o: main.c Initialize_domain.h savetimestep.h
	$(MPICC) $(DEBUG) $(LINK_OPS) main.c -o main.o $(CUDA_INC_PATH)

main_gpu.o: main_gpu.cu savetimestep.h gpu_kernels.cu
	$(NVCC) $(DEBUG) $(LINK_OPS) main_gpu.cu -o main_gpu.o $(MPI_INC_PATH)

clean:
	rm $(MPI_CUDA) $(OBJECTS) 
# To remove the data files
remove:
	rm -rf result/*.dat nohup.out
# To remove the generated images (after post processing)
remove_images:
	rm -rf result/*.png
	
	
