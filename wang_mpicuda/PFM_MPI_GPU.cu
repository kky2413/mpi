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

#include "savetimestep.cu"  // output파일들을 생성하는 코드 include
// Include code for creating output files

#define PI acos(-1.0)  // PI 상수 정의
// Define PI constant
#define Tolerance 1.0e-10  // 허용 오차 정의
// Define tolerance
#define COMPERR 1.0e-6  // 비교 오류 정의
// Define comparison error

#include "kernels_cuda.cu"  // GPU 커널 함수 include
// Include GPU kernel functions

// GPU 함수 정의
// Define GPU function
extern "C" void pfm_mpi_gpu(double *phi, double *comp, int nx, int ny, int np, int ni, char *argv[], int P, double del_x, double del_y, double del_t, double Acnst, double Bcnst, double kappaC, 
double kappaPhi, double matcomp, double partcomp, double *vtotX_d, double *vtotY_d, double *velX_d, double *velY_d, double emob, double *sum_phi_d, double *sumsq_phi_d, double *sumcub_phi_d, double Dvol, double Dvap, double Dsurf, double Dgb, double gsize, int op_steps, int num_steps, int threadsPerBlock_X, int threadsPerBlock_Y, int nprocs, int top, int bottom, int Nx, int Ny, int rank) {
  
  FILE *fw;
  char NAME[950], hostname[HOST_NAME_MAX + 1];

  int i, j, k, l;

  // GPU 메모리 포인터 선언
  // Declare GPU memory pointers
  double *phi_old_d, *phi_new_d;
  int *eta_d, *num_phi_d, *zeta_d;
  double *comp_d, *comp_new_d, *uc_d, *mu_d, *fen_d, *dummy_d;
  double *sum_uc_d, *sum_fen_d, *lamda_d;
  double *dphi_d;
  double *Diff, *fenergy, *Ceq;
  double *diff_d;
  double *Dvol_d, *Dvap_d, *Dsur_d, *Dgbd_d;
  int *eta;

  // 추가적인 GPU 메모리 포인터 선언
  // Declare additional GPU memory pointers
  double *dforX_d, *dforY_d, *dtor_d;
  double *forcX_d, *forcY_d, *torq_d, *radcX_d, *radcY_d, *vol_d;
  double kappaRbm, phicoff, compcoff, Mtra, Mrot;

  int blocks, blocks_xy;
  int threadsPerBlock, phinx, phiny, compnx, compny;
  int system_size;
  int count;

  int initcount, flag;

  // MPI 통신을 위한 버퍼 선언
  // Declare buffers for MPI communication
  double *buff_send_0;
  double *buff_send_n; 
  double *buff_recv_0;
  double *buff_recv_n;
  double *buff_send_p0;
  double *buff_send_pn; 
  double *buff_recv_p0;
  double *buff_recv_pn;

  clock_t gpu_start, gpu_end;

  double emtj, sigma, delta, tifac, garea, tot_area;
  double pss, www, cep, emm;
  int wid;

  double q_part;
  double Dpart, Dvoid;
 
  int adv_flag, device_flag;
  
  // 재시작(restart) 관련 변수 선언
  // Declare variables related to restart
  int res_stat, res_steps;
  
  int nDevices;
  cudaError_t err; 
  
  int tag_0 = 10; 
  int tag_n = 20;
  
  // MPI 비동기 통신을 위한 요청(request)과 상태(status) 선언
  // Declare MPI requests and statuses for asynchronous communication
  MPI_Request sreq_0, sreq_n;
  MPI_Status status_0, status_n;
  gpu_start = clock();

  // 현재 랭크의 호스트 이름 가져오기
  // Get the hostname of the current rank
  gethostname(hostname, HOST_NAME_MAX + 1);

  nDevices = -1;
  
  // 사용 가능한 CUDA 디바이스 수 확인
  // Check the number of available CUDA devices
  cudaGetDeviceCount(&nDevices);
  if ( nDevices <= 0 ) { 
    printf("ERROR: No .of Devices on this host = 0 for rank = %d\n", rank); 
    printf("Exiting code\n");
    exit(2);
  }
  
  // CUDA 장치를 리셋하고 랭크에 맞는 장치 설정
  // Reset CUDA device and set device according to rank
  cudaDeviceReset();
  cudaSetDevice(rank%nDevices);
  printf("Rank = %d attached to DeviceId = %d on host = %s\n", rank, rank%nDevices, hostname);
  MPI_Barrier(MPI_COMM_WORLD);

  // MPI 통신을 위한 버퍼의 메모리 할당
  // Allocate memory for buffers for MPI communication
  buff_send_0 = (double*)malloc(ny*sizeof(double));
  buff_send_n = (double*)malloc(ny*sizeof(double)); 
  buff_recv_0 = (double*)malloc(ny*sizeof(double));
  buff_recv_n = (double*)malloc(ny*sizeof(double));
  buff_send_p0 = (double*)malloc(ni*ny*sizeof(double));
  buff_send_pn = (double*)malloc(ni*ny*sizeof(double)); 
  buff_recv_p0 = (double*)malloc(ni*ny*sizeof(double));
  buff_recv_pn = (double*)malloc(ni*ny*sizeof(double));
  
  // CUDA 블록 수 계산
  // Calculate number of CUDA blocks
  blocks = ceil((double) nx*ny*ni / 256);
  blocks_xy = ceil((double) nx*ny / 256);
  threadsPerBlock_X = 16;
  threadsPerBlock_Y = 16;
  threadsPerBlock = 256;
  phinx = ceil((double) nx / 8);
  phiny = ceil((double) ny / 8);
  compnx = ceil((double) nx / 32);
  compny = ceil((double) ny / 32);
  
  // GPU 메모리 할당
  // Allocate GPU memory
  CHECK(cudaMalloc ((void**)&phi_old_d,sizeof(double)*nx*ny*ni));
  CHECK(cudaMalloc ((void**)&phi_new_d,sizeof(double)*nx*ny*ni));
  CHECK(cudaMalloc ((void**)&sum_phi_d,sizeof(double)*nx*ny));
  CHECK(cudaMalloc ((void**)&sumsq_phi_d,sizeof(double)*nx*ny));
  CHECK(cudaMalloc ((void**)&sumcub_phi_d,sizeof(double)*nx*ny));

  CHECK(cudaMalloc ((void**)&dforX_d,sizeof(double)*nx*ny*ni));
  CHECK(cudaMalloc ((void**)&dforY_d,sizeof(double)*nx*ny*ni));
  CHECK(cudaMalloc ((void**)&dtor_d,sizeof(double)*nx*ny*ni));
  CHECK(cudaMalloc ((void**)&velX_d,sizeof(double)*nx*ny*ni));
  CHECK(cudaMalloc ((void**)&velY_d,sizeof(double)*nx*ny*ni));

  CHECK(cudaMalloc ((void**)&forcX_d,sizeof(double)*ni));
  CHECK(cudaMalloc ((void**)&forcY_d,sizeof(double)*ni));
  CHECK(cudaMalloc ((void**)&torq_d,sizeof(double)*ni));
  CHECK(cudaMalloc ((void**)&radcX_d,sizeof(double)*ni));
  CHECK(cudaMalloc ((void**)&radcY_d,sizeof(double)*ni));
  CHECK(cudaMalloc ((void**)&vol_d,sizeof(double)*ni));

  CHECK(cudaMalloc ((void**)&vtotX_d,sizeof(double)*nx*ny));
  CHECK(cudaMalloc ((void**)&vtotY_d,sizeof(double)*nx*ny));

  CHECK(cudaMalloc ((void**)&comp_d,sizeof(double)*nx*ny));
  CHECK(cudaMalloc ((void**)&comp_new_d,sizeof(double)*nx*ny));
  CHECK(cudaMalloc ((void**)&mu_d,sizeof(double)*nx*ny));
  CHECK(cudaMalloc ((void**)&diff_d,sizeof(double)*nx*ny));

  CHECK(cudaMalloc ((void**)&Dvol_d,sizeof(double)*nx*ny));
  CHECK(cudaMalloc ((void**)&Dvap_d,sizeof(double)*nx*ny));
  CHECK(cudaMalloc ((void**)&Dsur_d,sizeof(double)*nx*ny));
  CHECK(cudaMalloc ((void**)&Dgbd_d,sizeof(double)*nx*ny));

  // 초기 데이터를 GPU로 복사
  // Copy initial data to GPU
  CHECK(cudaMemcpy(phi_new_d,phi,sizeof(double)*nx*ny*ni,cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(comp_new_d,comp,sizeof(double)*nx*ny,cudaMemcpyHostToDevice));

    if ( rank == 0 ) {
    sprintf(NAME,"paramters_gpu_For_%s", argv[1]);
    fw = fopen(NAME,"w");
    //time(&tim);
    fprintf(fw, "Name of input file: %s\n\n", argv[1]);
    //fprintf(fw, "Simulation date and time: %s\n", ctime(&tim));
    fprintf(fw, "Nx = %d\n", nx);
    fprintf(fw, "Ny = %d\n", ny);
    fprintf(fw, "ni = %d\n", ni);
    fprintf(fw, "del_x = %le\n", del_x);
    fprintf(fw, "del_y = %le\n", del_y);
    fprintf(fw, "del_t = %le\n", del_t);
    fprintf(fw, "Block size = %d\n", blocks);
    fprintf(fw, "blocks_xy = %d\n", blocks_xy);
    fprintf(fw, "Threads per block (x)  = %d\n", threadsPerBlock_X);
    fprintf(fw, "Threads per block (y)  = %d\n", threadsPerBlock_Y);
    fprintf(fw, "phinx = %d\n", phinx);
    fprintf(fw, "phiny = %d\n", phiny);
    //fprintf(fw, "Seed = %ld\n", SEED);
    fprintf(fw, "Output frequency = %d\n", op_steps);
    fprintf(fw, "num_steps = %d\n", num_steps);
    fprintf(fw, "matcomp = %lf\n", matcomp);
    fprintf(fw, "partcomp = %lf\n", partcomp);
    fprintf(fw, "Acnst = %le\n", Acnst);
    fprintf(fw, "Bcnst = %le\n", Bcnst);
    fprintf(fw, "kappaC = %le\n", kappaC);
    fprintf(fw, "kappaPhi = %le\n", kappaPhi);
    fprintf(fw, "GB mobility = %le\n", emob);
    fprintf(fw, "Dvol = %le\n", Dvol);
    fprintf(fw, "Dvap = %le\n", Dvap);
    fprintf(fw, "Dsurf = %le\n", Dsurf);
    fprintf(fw, "Dgb = %le\n", Dgb);
    fprintf(fw, "Grain size = %lf\n", gsize);
    fprintf(fw, "kappaRbm = %lf\n", kappaRbm);
    fprintf(fw, "phicoff = %lf\n", phicoff);
    fprintf(fw, "compcoff = %lf\n", compcoff);
    fprintf(fw, "Mtra = %lf\n", Mtra);
    fprintf(fw, "Mrot = %lf\n", Mrot);
    fprintf(fw, "restart status = %d\n", res_stat);
    fprintf(fw, "restart steps = %d\n", res_steps);
    fprintf(fw, "advect flag = %d\n", adv_flag);
    fprintf(fw, "device flag = %d\n", device_flag);
    fclose(fw);
    }

  if ( rank == 0 ) printf("Starting time loop\n");
  for(P = 1; P <= num_steps; P++) {

    // GPU 커널 호출: 합계 계산
    // GPU kernel call: compute sums
    sharedSum<<<blocks_xy, threadsPerBlock, threadsPerBlock * sizeof(double)>>>(phi_new_d, sum_phi_d, ni);
    CHECK_KERNEL();
    sharedSumsq<<<blocks_xy, threadsPerBlock, threadsPerBlock * sizeof(double)>>>(phi_new_d, sumsq_phi_d, ni);
    CHECK_KERNEL();
    sharedSumcub<<<blocks_xy, threadsPerBlock, threadsPerBlock * sizeof(double)>>>(phi_new_d, sumcub_phi_d, ni);
    CHECK_KERNEL();

    // 초기 조건 설정을 위한 커널 호출
    // GPU kernel call to set initial conditions
    kernelIntPhi<<<dim3(blocks,1,1),dim3(threadsPerBlock_X,threadsPerBlock_Y,1)>>>(phi_new_d, phi_old_d, nx, ny,ni);
    CHECK_KERNEL();
    kernelIntcomp<<<dim3(blocks,1,1),dim3(threadsPerBlock_X,threadsPerBlock_Y,1)>>>(comp_new_d, comp_d, nx, ny);
    CHECK_KERNEL();
    MPI_Barrier(MPI_COMM_WORLD);

    kernelUpdPhi<<<dim3(phinx,phiny,ni),dim3(8,8,1)>>>(phi_new_d, phi_old_d, velX_d, velY_d, comp_d, sumsq_phi_d, nx, ny, ni, del_x, del_y, del_t, Bcnst, kappaPhi, emob, rank);
    CHECK_KERNEL();

    cudaMemcpy(buff_send_p0, &phi_new_d[0 + ni * (0 + (1)*ny)],    sizeof(double)*ni*ny, cudaMemcpyDeviceToHost);
    
    MPI_Isend(&buff_send_p0[0], ni*ny, MPI_DOUBLE,    top, tag_0, MPI_COMM_WORLD, &sreq_0);
    MPI_Recv (&buff_recv_pn[0], ni*ny, MPI_DOUBLE, bottom, tag_0, MPI_COMM_WORLD, &status_0);
    MPI_Wait(&sreq_0, &status_0);
    
    cudaMemcpy(&phi_new_d[0 + ni * (0 + (nx-1)*ny)], buff_recv_pn, sizeof(double)*ni*ny, cudaMemcpyHostToDevice);
    
    MPI_Barrier(MPI_COMM_WORLD);

    cudaMemcpy(buff_send_pn, &phi_new_d[0 + ni * (0 + (nx-2)*ny)], sizeof(double)*ni*ny, cudaMemcpyDeviceToHost);
    
    MPI_Isend(&buff_send_pn[0], ni*ny, MPI_DOUBLE, bottom, tag_n, MPI_COMM_WORLD, &sreq_n);
    MPI_Recv (&buff_recv_p0[0], ni*ny, MPI_DOUBLE,    top, tag_n, MPI_COMM_WORLD, &status_n);
    MPI_Wait(&sreq_n, &status_n);

    cudaMemcpy(&phi_new_d[0 + ni * (0 + (0)*ny)], buff_recv_p0, sizeof(double)*ni*ny, cudaMemcpyHostToDevice);
    
    MPI_Barrier(MPI_COMM_WORLD);

    // mu를 계산하는 커널 호출
    // GPU kernel call to calculate mu
    kernelCalcMu<<<dim3(compnx,compny,1),dim3(32,32,1)>>>(mu_d, comp_d, sumsq_phi_d, sumcub_phi_d, nx, ny, del_x, del_y, Acnst, Bcnst, kappaC, rank);
    CHECK_KERNEL();

    // 상위 랭크로부터 데이터를 보내고 받음
    // Send and receive mu data to and from upper rank
    cudaMemcpy(buff_send_0, &mu_d[0 + (1)*ny], sizeof(double)*ny, cudaMemcpyDeviceToHost);
    
    MPI_Isend(&buff_send_0[0], ny, MPI_DOUBLE, top, tag_0, MPI_COMM_WORLD, &sreq_0);
    MPI_Recv(&buff_recv_n[0], ny, MPI_DOUBLE, bottom, tag_0, MPI_COMM_WORLD, &status_0);
    MPI_Wait(&sreq_0, &status_0);
    
    cudaMemcpy(&mu_d[0 + (nx-1)*ny], buff_recv_n, sizeof(double)*ny, cudaMemcpyHostToDevice);
    
    MPI_Barrier(MPI_COMM_WORLD);

    cudaMemcpy(buff_send_n, &mu_d[0 + (nx-2)*ny], sizeof(double)*ny, cudaMemcpyDeviceToHost);
    
    MPI_Isend(&buff_send_n[0], ny, MPI_DOUBLE, bottom, tag_n, MPI_COMM_WORLD, &sreq_n);
    MPI_Recv(&buff_recv_0[0], ny, MPI_DOUBLE, top, tag_n, MPI_COMM_WORLD, &status_n);
    MPI_Wait(&sreq_n, &status_n);

    cudaMemcpy(&mu_d[0 + (0)*ny], buff_recv_0, sizeof(double)*ny, cudaMemcpyHostToDevice);
    
    MPI_Barrier(MPI_COMM_WORLD);

    // Diffusion 계산을 위한 커널 호출
    // GPU kernel call to calculate diffusion
    kernelCalcDiff<<<dim3(compnx,compny,1),dim3(32,32,1)>>>(diff_d, Dvol_d, Dvap_d, Dsur_d, Dgbd_d, comp_d, sum_phi_d, sumsq_phi_d, ny, Dvol, Dvap, Dsurf, Dgb, rank);
    CHECK_KERNEL();

    // 상위 랭크로부터 데이터를 보내고 받음
    // Send and receive diffusion data to and from upper rank
    cudaMemcpy(buff_send_0, &diff_d[0 + (1)*ny], sizeof(double)*ny, cudaMemcpyDeviceToHost);
    
    MPI_Isend(&buff_send_0[0], ny, MPI_DOUBLE, top, tag_0, MPI_COMM_WORLD, &sreq_0);
    MPI_Recv(&buff_recv_n[0], ny, MPI_DOUBLE, bottom, tag_0, MPI_COMM_WORLD, &status_0);
    MPI_Wait(&sreq_0, &status_0);
    
    cudaMemcpy(&diff_d[0 + (nx-1)*ny], buff_recv_n, sizeof(double)*ny, cudaMemcpyHostToDevice);
    
    MPI_Barrier(MPI_COMM_WORLD);

    cudaMemcpy(buff_send_n, &diff_d[0 + (nx-2)*ny], sizeof(double)*ny, cudaMemcpyDeviceToHost);
    
    MPI_Isend(&buff_send_n[0], ny, MPI_DOUBLE, bottom, tag_n, MPI_COMM_WORLD, &sreq_n);
    MPI_Recv(&buff_recv_0[0], ny, MPI_DOUBLE, top, tag_n, MPI_COMM_WORLD, &status_n);
    MPI_Wait(&sreq_n, &status_n);

    cudaMemcpy(&diff_d[0 + (0)*ny], buff_recv_0, sizeof(double)*ny, cudaMemcpyHostToDevice);
    
    MPI_Barrier(MPI_COMM_WORLD);

    // 농도를 업데이트하는 커널 호출
    // GPU kernel call to update composition
    kernelUpdComp<<<dim3(compnx,compny,1),dim3(32,32,1)>>>(comp_new_d, comp_d, vtotX_d, vtotY_d, mu_d, diff_d, nx, ny, del_x, del_y, del_t, rank);
    CHECK_KERNEL();

    // 상위 랭크로부터 데이터를 보내고 받음
    // Send and receive composition data to and from upper rank
    cudaMemcpy(buff_send_0, &comp_new_d[0 + (1)*ny], sizeof(double)*ny, cudaMemcpyDeviceToHost);
    
    MPI_Isend(&buff_send_0[0], ny, MPI_DOUBLE, top, tag_0, MPI_COMM_WORLD, &sreq_0);
    MPI_Recv(&buff_recv_n[0], ny, MPI_DOUBLE, bottom, tag_0, MPI_COMM_WORLD, &status_0);
    MPI_Wait(&sreq_0, &status_0);
    
    cudaMemcpy(&comp_new_d[0 + (nx-1)*ny], buff_recv_n, sizeof(double)*ny, cudaMemcpyHostToDevice);
    
    MPI_Barrier(MPI_COMM_WORLD);

    cudaMemcpy(buff_send_n, &comp_new_d[0 + (nx-2)*ny], sizeof(double)*ny, cudaMemcpyDeviceToHost);
    
    MPI_Isend(&buff_send_n[0], ny, MPI_DOUBLE, bottom, tag_n, MPI_COMM_WORLD, &sreq_n);
    MPI_Recv(&buff_recv_0[0], ny, MPI_DOUBLE, top, tag_n, MPI_COMM_WORLD, &status_n);
    MPI_Wait(&sreq_n, &status_n);

    cudaMemcpy(&comp_new_d[0 + (0)*ny], buff_recv_0, sizeof(double)*ny, cudaMemcpyHostToDevice);
    
    MPI_Barrier(MPI_COMM_WORLD);

    // 주기적으로 타임스텝을 저장
    // Save time step periodically
    if( P > 0 && P%op_steps == 0 ) {
      
      CHECK(cudaMemcpy(phi, phi_old_d,sizeof(double)*nx*ny*ni,cudaMemcpyDeviceToHost));
      CHECK(cudaMemcpy(comp, comp_d,sizeof(double)*nx*ny,cudaMemcpyDeviceToHost));

      Output_Conf(phi, comp, nx, ny, ni, P, argv, rank);

      if ( rank == 0 ) printf("Written time step: %d \n", P);
      
    }

    MPI_Barrier(MPI_COMM_WORLD);
  
  }

  // GPU 메모리 해제
  // Free GPU memory
  cudaFree (comp_d);
  cudaFree (comp_new_d);
  cudaFree (mu_d);
  cudaFree (diff_d);

  cudaFree (vtotX_d);
  cudaFree (vtotY_d);

  cudaFree (forcX_d);
  cudaFree (forcY_d);
  cudaFree (torq_d);
  cudaFree (radcX_d);
  cudaFree (radcY_d);
  cudaFree (vol_d);

  cudaFree (dforX_d);
  cudaFree (dforY_d);
  cudaFree (dtor_d);
  cudaFree (velX_d);
  cudaFree (velY_d);

  cudaFree (Dvol_d);
  cudaFree (Dvap_d);
  cudaFree (Dsur_d);
  cudaFree (Dgbd_d);

  cudaFree (phi_old_d);
  cudaFree (phi_new_d);
  cudaFree (sum_phi_d);
  cudaFree (sumsq_phi_d);
  cudaFree (sumcub_phi_d);

  // CPU 메모리 해제
  // Free CPU memory
  free(buff_send_0);
  free(buff_send_n);
  free(buff_recv_0);
  free(buff_recv_n);
  free(buff_send_p0);
  free(buff_send_pn);
  free(buff_recv_p0);
  free(buff_recv_pn);

gpu_end = clock();
printf("GPU 덧셈 연산 소요 시간 : %4.6f \n",
    (double)((double)(gpu_end - gpu_start) / CLOCKS_PER_SEC));

}




