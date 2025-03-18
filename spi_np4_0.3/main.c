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

#include "mpi.h"

#include "savetimestep.h"
#include "Initialize_domain.h"

// 다양한 함수 선언
// Declare various functions
void gpu(double *phi, double tfac,double alpha,double beta, int P,int threadsPerBlock, int threadsPerBlock_X, int threadsPerBlock_Y, int count, int op_steps, long num_steps, double del_x, double del_y, double del_t, int nx, int ny, int dnum, double alloycomp, double Acnst, double kappa, double noise, double atmob, char *argv[], int nprocs, int top, int bottom, int Nx, int Ny, int rank);
void Initialize_domain(double *phi, int radius, double del_x, int nx, int ny, int Nx, int Ny, int rank, long SEED, double alloycomp, double noise);
void savetimestep(double *phi, int nx, int ny, int P, int rank, char *argv[]);

int main(int argc, char * argv[]) {

  FILE *fw, *fr;
  char NAME[950], hostname[HOST_NAME_MAX + 1], param[100];
  
  int P;

  long SEED;

  double tfac, alpha, beta;

  double *phi_old_d, *phi_new_d, *mu_d;
  double *phi;

  int blocks_X, blocks_Y;
  int threadsPerBlock, threadsPerBlock_X, threadsPerBlock_Y;
  int count, op_steps;
  long num_steps;
  double del_x, del_y, del_t;
  double sim_time, total_time;
  int nx, ny, dnum, radius;
  int Nx, Ny;
  int dflag;
  double alloycomp;
  double Acnst, kappa;
  double noise, atmob;
  
  int rank, top, bottom, nprocs;
  int tag_0 = 10; 
  int tag_n = 20;
  MPI_Request sreq_0, sreq_n;
  MPI_Status status_0, status_n; 

  time_t tim;
  
  // MPI 초기화
  // Initialize MPI
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if ( rank == 0 ) printf("No. of processors in execution %d\n", nprocs);
  
  // 현재 호스트 이름 가져오기
  // Get current hostname
  gethostname(hostname, HOST_NAME_MAX + 1);
  printf("Host name: %s for rank = %d\n\n", hostname, rank);
 
  // 입력 파일 열기
  // Open input file
  fr = fopen(argv[1], "r");
  if (fr == NULL) {
    printf("ERROR: No input file \n \t or \n");
    printf("ERROR: file %s not found\n", argv[1]);
    printf("\nRunning Syntax:\n \t \tmpirun -np 1 ./spino <input_file> \n");
    printf("\nExiting\n");
    exit(2);
  }

  // 입력 파일에서 매개 변수 읽기
  // Read parameters from input file
  fscanf(fr, "%s%d",  param, &Nx);
  fscanf(fr, "%s%d",  param, &Ny);
  fscanf(fr, "%s%le", param, &del_x);
  fscanf(fr, "%s%le", param, &del_y);
  fscanf(fr, "%s%le", param, &tfac);
  fscanf(fr, "%s%d",  param, &threadsPerBlock_X);
  fscanf(fr, "%s%d",  param, &threadsPerBlock_Y);
  fscanf(fr, "%s%ld", param, &SEED);
  fscanf(fr, "%s%d",  param, &op_steps);
  fscanf(fr, "%s%ld", param, &num_steps);
  fscanf(fr, "%s%lf", param, &alloycomp);  
  fscanf(fr, "%s%le", param, &Acnst);
  fscanf(fr, "%s%le", param, &kappa);
  fscanf(fr, "%s%le", param, &atmob);
  fscanf(fr, "%s%d",  param, &dflag);
  fscanf(fr, "%s%lf", param, &noise);
  fscanf(fr, "%s%d", param, &dnum);
  fscanf(fr, "%s%d", param, &radius);
  fclose(fr);

  // 매개 변수 계산
  // Calculate parameters
  alpha = 2.0 * kappa * atmob;
  beta  = Acnst * atmob;
  del_t = tfac * del_x * del_x * del_x * del_x / ((16.0 * alpha + 4.0 * beta * del_x * del_x) * (double) dnum);

  if (rank == 0) printf("Read input parameters\n");

  // 각 프로세스의 도메인 크기 계산
  // Calculate domain size for each process
  nx = Nx / nprocs + 2;

  if (Nx % nprocs != 0) { 
    printf("ERROR: Nx is not divisible by no. of processors (np). Choose them accordingly.\n");
    exit(1);
    printf("Exiting\n");
  }
  ny = Ny;
  
  // 사용한 파라미터를 파일에 저장
  // Save used parameters to a file
  if (rank == 0) {
    sprintf(NAME, "paramters_used_For_%s", argv[1]);
    fw = fopen(NAME, "w");
    fprintf(fw, "Nx = %d\n", Nx);
    fprintf(fw, "Ny = %d\n", Ny);
    fprintf(fw, "del_x = %le\n", del_x);
    fprintf(fw, "del_y = %le\n", del_y);
    fprintf(fw, "del_t = %le\n", del_t);
    fprintf(fw, "threadsPerBlock_X = %d\n", threadsPerBlock_X);
    fprintf(fw, "threadsPerBlock_Y = %d\n", threadsPerBlock_Y);
    fprintf(fw, "SEED = %ld\n", SEED);
    fprintf(fw, "Output steps = %d\n", op_steps);
    fprintf(fw, "Total steps = %ld\n", num_steps);
    fprintf(fw, "Composition = %lf\n", alloycomp);  
    fprintf(fw, "Bulk coefficient = %le\n", Acnst);
    fprintf(fw, "Grad coefficient = %le\n", kappa);
    fprintf(fw, "Atomic mobility = %le\n", atmob);
    fprintf(fw, "Device flag = %d\n", dflag);
    fprintf(fw, "Noise level = %lf\n", noise);
    fprintf(fw, "nx = %d\n", nx);
    fprintf(fw, "ny = %d\n", ny);
    fclose(fw);
  }
  
  MPI_Barrier(MPI_COMM_WORLD);
  
  // 각 프로세스의 상위 및 하위 이웃 프로세스 설정
  // Set top and bottom neighboring processes for each process
  top = rank - 1; 
  if (rank == 0) { 
    top = nprocs - 1;
  }
  bottom = rank + 1; 
  if (rank == nprocs - 1) { 
    bottom = 0;
  }
  printf("( top, rank, bottom ) = ( %d, %d, %d )\n", top, rank, bottom);

  // phi 배열 할당
  // Allocate phi array
  phi = (double *)malloc(sizeof(double) * nx * ny);
  MPI_Barrier(MPI_COMM_WORLD);
  
  // 초기 도메인 설정
  // Set up initial domain
  if (rank == 0) printf("\nSetting up initial domain:\n");
  Initialize_domain(phi, radius, del_x, nx, ny, Nx, Ny, rank, SEED, alloycomp, noise);
  MPI_Barrier(MPI_COMM_WORLD);

  // 초기 경계 조건 설정
  // Set initial boundary conditions
  MPI_Isend(&phi[0 + (1) * ny], ny, MPI_DOUBLE, top, tag_0, MPI_COMM_WORLD, &sreq_0);
  MPI_Recv(&phi[0 + (nx - 1) * ny], ny, MPI_DOUBLE, bottom, tag_0, MPI_COMM_WORLD, &status_0);
  MPI_Wait(&sreq_0, &status_0);

  MPI_Isend(&phi[0 + (nx - 2) * ny], ny, MPI_DOUBLE, bottom, tag_n, MPI_COMM_WORLD, &sreq_n);
  MPI_Recv(&phi[0], ny, MPI_DOUBLE, top, tag_n, MPI_COMM_WORLD, &status_n);
  MPI_Wait(&sreq_n, &status_n);

  MPI_Barrier(MPI_COMM_WORLD);

  // 초기 도메인 저장
  // Save initial domain
  if (rank == 0) printf("Writing initial domain\n");
  P = 0;
  savetimestep(phi, nx, ny, P, rank, argv);
  MPI_Barrier(MPI_COMM_WORLD);
  
  // GPU 함수 호출
  // Call GPU function
  gpu(phi, tfac, alpha, beta, P, threadsPerBlock, threadsPerBlock_X, threadsPerBlock_Y, count, op_steps, num_steps, del_x, del_y, del_t, nx, ny, dnum, alloycomp, Acnst, kappa, noise, atmob, argv, nprocs, top, bottom, Nx, Ny, rank);

  MPI_Barrier(MPI_COMM_WORLD);

  // 메모리 해제
  // Free memory
  free(phi);
  
  if (rank == 0) printf("\nCode execution has completed\n");
  if (rank == 0) printf("\nDone; time to say good-bye!\n\n");

  // MPI 종료
  // Finalize MPI
  MPI_Finalize();

}


