#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>
#include <sys/stat.h>
#include <unistd.h>
#include <limits.h>
#include <time.h>
#include <stdint.h>
#include <unistd.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include "check.h"  // CUDA 오류 확인 매크로
// CUDA error checking macro

#include "mpi.h"  // 병렬 처리를 위한 MPI 라이브러리
// MPI library for parallel processing

#include "savetimestep.h"  // output파일들을 생성하는 코드 include
// Include code for creating output files
#include "gpu_kernels.cu"  // GPU 커널 함수 include
// Include GPU kernel functions

// GPU 함수 정의
// Define GPU function
extern "C" void gpu(double *phi, double tfac, double alpha, int P, double beta, int threadsPerBlock, int threadsPerBlock_X, int threadsPerBlock_Y, int count, int op_steps, long num_steps, double del_x, double del_y, double del_t, int nx, int ny, int dnum, double alloycomp, double Acnst, double kappa, double noise, double atmob, char *argv[], int nprocs, int top, int bottom, int Nx, int Ny, int rank) {
  
  FILE *fw;
  char NAME[950], hostname[HOST_NAME_MAX + 1];

  int i, j;

  double *phi_old_d, *phi_new_d, *mu_d;

  // MPI 통신을 위한 버퍼
  // Buffers for MPI communication
  double *buff_send_0;
  double *buff_send_n; 
  double *buff_recv_0;
  double *buff_recv_n;
  clock_t gpu_start, gpu_end;

  int blocks_X, blocks_Y;
  
  int nDevices;
  cudaError_t err; 
  
  // tag_0와 tag_n은 MPI 메시지 태그
  // tag_0 and tag_n are MPI message tags
  int tag_0 = 10; 
  int tag_n = 20;
  // sreq_0와 sreq_n은 비동기 통신 요청을 저장하는 데 사용되는 MPI 요청 객체
  // sreq_0 and sreq_n are MPI request objects used to store asynchronous communication requests
  MPI_Request sreq_0, sreq_n;
  MPI_Status status_0, status_n;
  gpu_start = clock();

  // 현재 랭크의 호스트 이름 가져오기
  // Get the hostname of the current rank
  gethostname(hostname, HOST_NAME_MAX + 1);

  nDevices = -1;
  cudaGetDeviceCount(&nDevices);
  if ( nDevices <= 0 ) { 
    printf("ERROR: No. of Devices on this host = 0 for rank = %d\n", rank); 
    printf("Exiting code\n");
    exit(2);
  }
  
  // CUDA 장치를 리셋하고 랭크에 맞는 장치 설정
  // Reset CUDA device and set device according to rank
  cudaDeviceReset();
  cudaSetDevice(rank % nDevices);
  printf("Rank = %d attached to DeviceId = %d on host = %s\n", rank, rank % nDevices, hostname);
  MPI_Barrier(MPI_COMM_WORLD);

  // MPI 통신을 위한 버퍼의 메모리 할당
  // Allocate memory for buffers for MPI communication
  buff_send_0 = (double*)malloc(ny * sizeof(double));
  buff_send_n = (double*)malloc(ny * sizeof(double)); 
  buff_recv_0 = (double*)malloc(ny * sizeof(double));
  buff_recv_n = (double*)malloc(ny * sizeof(double));
  
  // CUDA 블록 수 계산
  // Calculate number of CUDA blocks
  blocks_X = ceil((double) nx / (double) threadsPerBlock_X);
  blocks_Y = ceil((double) ny / (double) threadsPerBlock_Y);
  printf("nx = %d, ny = %d, threadsPerBlock_X = %d, threadsPerBlock_Y = %d, blocks_X = %d, blocks_Y = %d\n", nx, ny, threadsPerBlock_X, threadsPerBlock_Y, blocks_X, blocks_Y);
  
  // GPU 메모리 할당
  // Allocate GPU memory
  CHECK(cudaMalloc((void**)&phi_old_d, sizeof(double) * nx * ny));
  CHECK(cudaMalloc((void**)&phi_new_d, sizeof(double) * nx * ny));
  CHECK(cudaMalloc((void**)&mu_d, sizeof(double) * nx * ny));
  
  // 초기 phi 데이터를 GPU로 복사
  // Copy initial phi data to GPU
  CHECK(cudaMemcpy(phi_new_d, phi, sizeof(double) * nx * ny, cudaMemcpyHostToDevice));
  
  // phi 값을 업데이트하는 커널 호출
  // Call kernel to update phi values
  kernelUpdPhi<<<dim3(blocks_X, blocks_Y, 1), dim3(threadsPerBlock_X, threadsPerBlock_Y, 1)>>>(phi_new_d, phi_old_d, nx, ny, rank);
  CHECK_KERNEL();
  
  // MPI 프로그램에서 동기화 지점을 설정
  // Set synchronization point in MPI program
  MPI_Barrier(MPI_COMM_WORLD);

  if (rank == 0) printf("Starting time loop\n");
  for (P = 1; P <= num_steps; P++) {

    //if (rank == 0) printf("Time-step: %d \n", P);
    
    // mu를 계산하는 커널 호출
    // Call kernel to calculate mu
    kernelCalcMu<<<dim3(blocks_X, blocks_Y, 1), dim3(threadsPerBlock_X, threadsPerBlock_Y, 1)>>>(mu_d, phi_old_d, nx, ny, del_x, del_y, Acnst, kappa, rank);
    CHECK_KERNEL();
    
    // 상위 랭크로부터 데이터를 보내고 받음
    // Send and receive mu data to and from upper rank
    cudaMemcpy(buff_send_0, &mu_d[0 + (1) * ny], sizeof(double) * ny, cudaMemcpyDeviceToHost);
    
    MPI_Isend(&buff_send_0[0], ny, MPI_DOUBLE, top, tag_0, MPI_COMM_WORLD, &sreq_0);
    MPI_Recv(&buff_recv_n[0], ny, MPI_DOUBLE, bottom, tag_0, MPI_COMM_WORLD, &status_0);
    MPI_Wait(&sreq_0, &status_0);
    
    cudaMemcpy(&mu_d[0 + (nx - 1) * ny], buff_recv_n, sizeof(double) * ny, cudaMemcpyHostToDevice);
    
    MPI_Barrier(MPI_COMM_WORLD);

    cudaMemcpy(buff_send_n, &mu_d[0 + (nx - 2) * ny], sizeof(double) * ny, cudaMemcpyDeviceToHost);
    
    MPI_Isend(&buff_send_n[0], ny, MPI_DOUBLE, bottom, tag_n, MPI_COMM_WORLD, &sreq_n);
    MPI_Recv(&buff_recv_0[0], ny, MPI_DOUBLE, top, tag_n, MPI_COMM_WORLD, &status_n);
    MPI_Wait(&sreq_n, &status_n);

    cudaMemcpy(&mu_d[0 + (0) * ny], buff_recv_0, sizeof(double) * ny, cudaMemcpyHostToDevice);
    
    MPI_Barrier(MPI_COMM_WORLD);

    // 새로운 phi 값을 계산하는 커널 호출
    // Call kernel to calculate new phi values
    kernelCalCom<<<dim3(blocks_X, blocks_Y, 1), dim3(threadsPerBlock_X, threadsPerBlock_Y, 1)>>>(mu_d, phi_new_d, phi_old_d, nx, ny, del_x, del_y, del_t, atmob, rank);
    CHECK_KERNEL();
    
    // 상위 랭크로부터 데이터를 보내고 받음
    // Send and receive phi data to and from upper rank
    cudaMemcpy(buff_send_0, &phi_new_d[0 + (1) * ny], sizeof(double) * ny, cudaMemcpyDeviceToHost);
    
    MPI_Isend(&buff_send_0[0], ny, MPI_DOUBLE, top, tag_0, MPI_COMM_WORLD, &sreq_0);
    MPI_Recv(&buff_recv_n[0], ny, MPI_DOUBLE, bottom, tag_0, MPI_COMM_WORLD, &status_0);
    MPI_Wait(&sreq_0, &status_0);
    
    cudaMemcpy(&phi_new_d[0 + (nx - 1) * ny], buff_recv_n, sizeof(double) * ny, cudaMemcpyHostToDevice);
    
    MPI_Barrier(MPI_COMM_WORLD);

    cudaMemcpy(buff_send_n, &phi_new_d[0 + (nx - 2) * ny], sizeof(double) * ny, cudaMemcpyDeviceToHost);
    
    MPI_Isend(&buff_send_n[0], ny, MPI_DOUBLE, bottom, tag_n, MPI_COMM_WORLD, &sreq_n);
    MPI_Recv(&buff_recv_0[0], ny, MPI_DOUBLE, top, tag_n, MPI_COMM_WORLD, &status_n);
    MPI_Wait(&sreq_n, &status_n);

    cudaMemcpy(&phi_new_d[0 + (0) * ny], buff_recv_0, sizeof(double) * ny, cudaMemcpyHostToDevice);
    
    MPI_Barrier(MPI_COMM_WORLD);

    // phi 값을 업데이트하는 커널 호출
    // Call kernel to update phi values
    kernelUpdPhi<<<dim3(blocks_X, blocks_Y, 1), dim3(threadsPerBlock_X, threadsPerBlock_Y, 1)>>>(phi_new_d, phi_old_d, nx, ny, rank);
    CHECK_KERNEL();

    // 주기적으로 타임스텝을 저장
    // Save time step periodically
    if (P > 0 && P % op_steps == 0) {
      
      CHECK(cudaMemcpy(phi, phi_old_d, sizeof(double) * nx * ny, cudaMemcpyDeviceToHost));

      savetimestep(phi, nx, ny, P, rank, argv);
      
      if (rank == 0) printf("Written time step: %d \n", P);

    }

    MPI_Barrier(MPI_COMM_WORLD);
  
  }
  
  // GPU 메모리 해제
  // Free GPU memory
  cudaFree(phi_old_d);
  cudaFree(mu_d);
  cudaFree(phi_new_d);
  
  // CPU 메모리 해제
  // Free CPU memory
  free(buff_send_0);
  free(buff_send_n);
  free(buff_recv_0);
  free(buff_recv_n);

gpu_end = clock();
printf("GPU 덧셈 연산 소요 시간 : %4.6f \n",
    (double)((double)(gpu_end - gpu_start) / CLOCKS_PER_SEC));

}

