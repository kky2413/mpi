#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>
#include<stddef.h>
#include <sys/stat.h>
#include <unistd.h>
#include <limits.h>
#include <time.h>
#include <stdint.h>
#include <unistd.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include "check.h" 

#include "mpi.h" 

#include "savetimestep.cu" 

#define PI acos(-1.0) 
#define Tolerance 1.0e-10
#define COMPERR 1.0e-6
#define nog 533

#include "kernels_cuda.cu"  

extern "C" void pfm_mpi_gpu (double *phi, int *gid, double *comp, int *nap_phi, int nx, int ny, int nz, int np, int ni, double alpha, double beta,
char *argv[], int P, double del_x, double del_y, double del_t, double del_z, double Acnst, double Bcnst, double kappaC, 
double kappaPhi, double matcomp, double partcomp, double emob, double *sum_phi_d, double *sumsq_phi_d, double *sumcub_phi_d, 
double Dvol, double Dvap, double Dsurf, double Dgb, double gsize, int op_steps, int num_steps, int threadsPerBlock_X, int threadsPerBlock_Y, 
int nprocs, int top, int bottom, int Nx, int Ny, int Nz,int rank, long SEED, int cmpres, int mixrat, double kappaRbm, double phicoff, 
double compcoff, double Mtra, double Mrot,   double pss, double www, double cep, double emm) {
  
  FILE *fw, *fpin, *fpcout;
  char NAME[950], hostname[HOST_NAME_MAX + 1];

  int i, j, h, k, l;

  double *phi_old_d, *phi_new_d;
  int *eta_d, *num_phi_d, *zeta_d;
  double *comp_d, *comp_new_d, *uc_d, *mu_d, *fen_d, *dummy_d;
  double *sum_uc_d, *sum_fen_d, *lamda_d;
  double *dphi_d;
  double *Diff, *fenergy, *Ceq;
  double *diff_d;
  double *Dvol_d, *Dvap_d, *Dsur_d, *Dgbd_d;
  int *eta;
  int *gid_d, *nap_phi_d;
  int numact;

  double *dforX_d, *dforY_d, *dtor_d;
  double *forcX_d, *forcY_d, *torq_d, *radcX_d, *radcY_d, *vol_d;

  int blocks, blocks_xy, blocks_xyz;
  int threadsPerBlock, phinx, phiny, compnx, compny;
  int system_size;
  int count;

  int initcount, flag;

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
  int wid;

  double q_part;
  double Dpart, Dvoid;
 
  int adv_flag, device_flag;

  int res_stat, res_steps;
  
  int nDevices;
  int device_map[] = {0,0};
  int num_devices = sizeof(device_map) / sizeof(device_map[0]);
  cudaError_t err; 
  int device_id = device_map[rank % num_devices];
  
  int tag_0 = 10; 
  int tag_n = 20;
  
  MPI_Request sreq_0, sreq_n;
  MPI_Status status_0, status_n;
  gpu_start = clock();

  gethostname(hostname, HOST_NAME_MAX + 1);

  nDevices = -1;
  cudaGetDeviceCount(&nDevices);
  if ( nDevices <= 0 ) { 
    printf("ERROR: No. of Devices on this host = 0 for rank = %d\n", rank); 
    printf("Exiting code\n");
    exit(2);
  }
  
  cudaDeviceReset();
  cudaSetDevice(device_id);
  printf("Rank = %d attached to DeviceId = %d on host = %s\n", rank, rank % nDevices, hostname);
  MPI_Barrier(MPI_COMM_WORLD);

  size_t free_mem, total_mem;
  cudaMemGetInfo(&free_mem, &total_mem);
  printf("Rank %d: Free memory: %zu, Total memory: %zu\n", rank, free_mem, total_mem);
  
  buff_send_0 = (double*)malloc(ny*nz*sizeof(double));
  buff_send_n = (double*)malloc(ny*nz*sizeof(double)); 
  buff_recv_0 = (double*)malloc(ny*nz*sizeof(double));
  buff_recv_n = (double*)malloc(ny*nz*sizeof(double));
  buff_send_p0 = (double*)malloc(ni*ny*nz*sizeof(double));
  buff_send_pn = (double*)malloc(ni*ny*nz*sizeof(double)); 
  buff_recv_p0 = (double*)malloc(ni*ny*nz*sizeof(double));
  buff_recv_pn = (double*)malloc(ni*ny*nz*sizeof(double));
  
  blocks = ceil((double) nx*ny*nz*ni / 256);
  blocks_xyz = ceil((double) nx*ny*nz / 256);
  threadsPerBlock_X = 16;
  threadsPerBlock_Y = 16;
  threadsPerBlock = 256;
  phinx = ceil((double) nx / 8);
  
  CHECK(cudaMalloc ((void**)&phi_old_d,sizeof(double)*nx*ny*nz*ni));
  CHECK(cudaMalloc ((void**)&phi_new_d,sizeof(double)*nx*ny*nz*ni));
  CHECK(cudaMalloc ((void**)&sum_phi_d,sizeof(double)*nx*ny*nz));
  CHECK(cudaMalloc ((void**)&sumsq_phi_d,sizeof(double)*nx*ny*nz));
  CHECK(cudaMalloc ((void**)&sumcub_phi_d,sizeof(double)*nx*ny*nz));
	  
  CHECK(cudaMalloc ((void**)&gid_d,sizeof(int)*nx*ny*nz*ni));    
  CHECK(cudaMalloc ((void**)&num_phi_d,sizeof(int)*nx*ny*nz));   
  CHECK(cudaMalloc ((void**)&nap_phi_d,sizeof(int)*nx*ny*nz));   

  CHECK(cudaMalloc ((void**)&comp_d,sizeof(double)*nx*ny*nz));
  CHECK(cudaMalloc ((void**)&comp_new_d,sizeof(double)*nx*ny*nz));
  CHECK(cudaMalloc ((void**)&mu_d,sizeof(double)*nx*ny*nz));
  CHECK(cudaMalloc ((void**)&diff_d,sizeof(double)*nx*ny*nz));

  CHECK(cudaMemcpy(gid_d, gid, sizeof(int) * nx * ny * nz * ni, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(phi_new_d, phi, sizeof(double) * nx * ny * nz * ni, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(nap_phi_d, nap_phi, sizeof(int) * nx * ny * nz, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(comp_new_d, comp, sizeof(double) * nx * ny * nz, cudaMemcpyHostToDevice));

  if ( rank == 0 ) printf("Starting time loop\n");
  for(P = 1; P <= num_steps; P++) {
    
    kernelIntPhi<<<dim3(blocks,1,1),dim3(threadsPerBlock_X,threadsPerBlock_Y,1)>>>(phi_old_d, nx, ny, nz, ni, rank);
    cudaDeviceSynchronize();
    sharedUpdList<<<blocks_xyz, threadsPerBlock, threadsPerBlock * sizeof(double)>>>(num_phi_d, nap_phi_d, gid_d, phi_new_d, phi_old_d, pss, ni, nx, ny, nz, rank);
    cudaDeviceSynchronize();
    MPI_Barrier(MPI_COMM_WORLD);
    kernelIntPhi<<<dim3(blocks,1,1),dim3(threadsPerBlock_X,threadsPerBlock_Y,1)>>>(phi_new_d, nx, ny, nz, ni, rank);
    cudaDeviceSynchronize();
    MPI_Barrier(MPI_COMM_WORLD);

    kernelUpdComp<<<dim3(blocks_xyz,1,1),dim3(threadsPerBlock_X,threadsPerBlock_Y,1)>>>(comp_new_d, comp_d, nx, ny, nz, rank);
    cudaDeviceSynchronize();
    MPI_Barrier(MPI_COMM_WORLD);

    sharedSum<<<blocks_xyz, threadsPerBlock, threadsPerBlock * sizeof(double)>>>(num_phi_d, phi_old_d, sum_phi_d, ni, pss);
    sharedSumsq<<<blocks_xyz, threadsPerBlock, threadsPerBlock * sizeof(double)>>>(num_phi_d, phi_old_d, sumsq_phi_d, ni, pss);
    sharedSumcub<<<blocks_xyz, threadsPerBlock, threadsPerBlock * sizeof(double)>>>(num_phi_d, phi_old_d, sumcub_phi_d, ni, pss);
    cudaDeviceSynchronize();

    sharedCalcPhi<<<blocks_xyz, threadsPerBlock, threadsPerBlock * sizeof(double)>>>(num_phi_d, nap_phi_d, gid_d, phi_new_d, phi_old_d, comp_d, sumsq_phi_d, nx, ny, nz, np, ni, del_x, del_y, del_z, del_t, Bcnst, kappaPhi, emob, rank);
    CHECK_KERNEL();

    cudaMemcpy(buff_send_p0, &phi_new_d[0 + (0 + nz * (0 + (1) * ny)) * ni],    sizeof(double)*ni*ny*nz, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    MPI_Isend(&buff_send_p0[0], ni*ny*nz, MPI_DOUBLE,    top, tag_0, MPI_COMM_WORLD, &sreq_0);
    MPI_Recv (&buff_recv_pn[0], ni*ny*nz, MPI_DOUBLE, bottom, tag_0, MPI_COMM_WORLD, &status_0);
    MPI_Wait(&sreq_0, &status_0);
    
    cudaMemcpy(&phi_new_d[0 + (0 + nz * (0 + (nx-1) * ny)) * ni], buff_recv_pn, sizeof(double)*ni*ny*nz, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    
    MPI_Barrier(MPI_COMM_WORLD);

    cudaMemcpy(buff_send_pn, &phi_new_d[0 + (0 + nz * (0 + (nx-2) * ny)) * ni], sizeof(double)*ni*ny*nz, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    
    MPI_Isend(&buff_send_pn[0], ni*ny*nz, MPI_DOUBLE, bottom, tag_n, MPI_COMM_WORLD, &sreq_n);
    MPI_Recv (&buff_recv_p0[0], ni*ny*nz, MPI_DOUBLE,    top, tag_n, MPI_COMM_WORLD, &status_n);
    MPI_Wait(&sreq_n, &status_n);

    cudaMemcpy(&phi_new_d[0 + (0 + nz * (0 + 0 * ny)) * ni], buff_recv_p0, sizeof(double)*ni*ny*nz, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    
    MPI_Barrier(MPI_COMM_WORLD);

    kernelCalcMu<<<dim3(phinx,ny/8,nz/8),dim3(8,8,8)>>>(mu_d, comp_d, sumsq_phi_d, sumcub_phi_d, nx, ny, nz, del_x, del_y, del_z, Acnst, Bcnst, kappaC, rank);
    CHECK_KERNEL();
    cudaDeviceSynchronize();
    MPI_Barrier(MPI_COMM_WORLD);

    cudaMemcpy(buff_send_0, &mu_d[0 + nz * (0+(1)*ny)], sizeof(double)*ny*nz, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    MPI_Isend(&buff_send_0[0], ny*nz, MPI_DOUBLE, top, tag_0, MPI_COMM_WORLD, &sreq_0);
    MPI_Recv(&buff_recv_n[0], ny*nz, MPI_DOUBLE, bottom, tag_0, MPI_COMM_WORLD, &status_0);
    MPI_Wait(&sreq_0, &status_0);
    
    cudaMemcpy(&mu_d[0 + nz * (0+(nx-1)*ny)], buff_recv_n, sizeof(double)*ny*nz, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    
    MPI_Barrier(MPI_COMM_WORLD);

    cudaMemcpy(buff_send_n, &mu_d[0 + nz * (0+(nx-2)*ny)], sizeof(double)*ny*nz, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    
    MPI_Isend(&buff_send_n[0], ny*nz, MPI_DOUBLE, bottom, tag_n, MPI_COMM_WORLD, &sreq_n);
    MPI_Recv(&buff_recv_0[0], ny*nz, MPI_DOUBLE, top, tag_n, MPI_COMM_WORLD, &status_n);
    MPI_Wait(&sreq_n, &status_n);

    cudaMemcpy(&mu_d[0 + nz * (0+0*ny)], buff_recv_0, sizeof(double)*ny*nz, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    
    MPI_Barrier(MPI_COMM_WORLD);

    kernelCalcDiff<<<dim3(phinx,ny/8,nz/8),dim3(8,8,8)>>>(diff_d, comp_d, sum_phi_d, sumsq_phi_d, ny, nz, nx, Dvol, Dvap, Dsurf, Dgb, rank);
    CHECK_KERNEL();
    cudaDeviceSynchronize();

    cudaMemcpy(buff_send_0, &diff_d[0 + nz * (0+(1)*ny)], sizeof(double)*ny*nz, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    
    MPI_Isend(&buff_send_0[0], ny*nz, MPI_DOUBLE, top, tag_0, MPI_COMM_WORLD, &sreq_0);
    MPI_Recv(&buff_recv_n[0], ny*nz, MPI_DOUBLE, bottom, tag_0, MPI_COMM_WORLD, &status_0);
    MPI_Wait(&sreq_0, &status_0);
    
    cudaMemcpy(&diff_d[0 + nz * (0+(nx-1)*ny)], buff_recv_n, sizeof(double)*ny*nz, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    
    MPI_Barrier(MPI_COMM_WORLD);

    cudaMemcpy(buff_send_n, &diff_d[0 + nz * (0+(nx-2)*ny)], sizeof(double)*ny*nz, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    
    MPI_Isend(&buff_send_n[0], ny*nz, MPI_DOUBLE, bottom, tag_n, MPI_COMM_WORLD, &sreq_n);
    MPI_Recv(&buff_recv_0[0], ny*nz, MPI_DOUBLE, top, tag_n, MPI_COMM_WORLD, &status_n);
    MPI_Wait(&sreq_n, &status_n);

    cudaMemcpy(&diff_d[0 + nz * (0+0*ny)], buff_recv_0, sizeof(double)*ny*nz, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    
    MPI_Barrier(MPI_COMM_WORLD);

    kernelCalcComp<<<dim3(phinx,ny/8,nz/8),dim3(8,8,8)>>>(comp_new_d, comp_d, mu_d, diff_d, nx, ny, nz, del_x, del_y, del_z, del_t, rank);
    CHECK_KERNEL();
    cudaDeviceSynchronize();
    MPI_Barrier(MPI_COMM_WORLD);

    cudaMemcpy(buff_send_0, &comp_new_d[0 + nz * (0+(1)*ny)], sizeof(double)*ny*nz, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    
    MPI_Isend(&buff_send_0[0], ny*nz, MPI_DOUBLE, top, tag_0, MPI_COMM_WORLD, &sreq_0);
    MPI_Recv(&buff_recv_n[0], ny*nz, MPI_DOUBLE, bottom, tag_0, MPI_COMM_WORLD, &status_0);
    MPI_Wait(&sreq_0, &status_0);
    
    cudaMemcpy(&comp_new_d[0 + nz * (0+(nx-1)*ny)], buff_recv_n, sizeof(double)*ny*nz, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    
    MPI_Barrier(MPI_COMM_WORLD);

    cudaMemcpy(buff_send_n, &comp_new_d[0 + nz * (0+(nx-2)*ny)], sizeof(double)*ny*nz, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    
    MPI_Isend(&buff_send_n[0], ny*nz, MPI_DOUBLE, bottom, tag_n, MPI_COMM_WORLD, &sreq_n);
    MPI_Recv(&buff_recv_0[0], ny*nz, MPI_DOUBLE, top, tag_n, MPI_COMM_WORLD, &status_n);
    MPI_Wait(&sreq_n, &status_n);

    cudaMemcpy(&comp_new_d[0 + nz * (0+0*ny)], buff_recv_0, sizeof(double)*ny*nz, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    
    MPI_Barrier(MPI_COMM_WORLD);

    if( P > 0 && P%op_steps == 0 ) {
      
      cudaMemcpy(phi, phi_old_d, sizeof(double) * nx * ny *nz* ni, cudaMemcpyDeviceToHost);
      cudaMemcpy(gid, gid_d, sizeof(int) * nx * ny * nz * ni, cudaMemcpyDeviceToHost); 
      cudaMemcpy(comp, comp_d, sizeof(double) * nx * ny * nz, cudaMemcpyDeviceToHost);
      cudaMemcpy(nap_phi, num_phi_d, sizeof(int) * nx * ny * nz, cudaMemcpyDeviceToHost); 

      Output_Conf(phi, gid, comp, nap_phi, nx, ny, nz, ni, P, rank, nprocs, argv, Nx);
    
      if ( rank == 0 ) printf("Written time step: %d \n", P);
      
    }

    MPI_Barrier(MPI_COMM_WORLD);
  
  }

  cudaFree (comp_d);
  cudaFree (comp_new_d);
  cudaFree (mu_d);
  cudaFree (diff_d);

  cudaFree (phi_old_d);
  cudaFree (phi_new_d);
  cudaFree (sum_phi_d);
  cudaFree (sumsq_phi_d);
  cudaFree (sumcub_phi_d);

  cudaFree (num_phi_d);                
  cudaFree (nap_phi_d);                
  cudaFree (gid_d);

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

