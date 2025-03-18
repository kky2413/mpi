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

#include "savetimestep.cu"  // output���ϵ��� �����ϴ� �ڵ� include
// Include code for creating output files
#include "Initialize_domain.cu"  // �ʱ� �������� �����ϴ� �ڵ� include
// Include code for initializing the domain

// �پ��� �Լ� ����
// Declare various functions
void pfm_mpi_gpu(double *phi, double *comp, int nx, int ny, int np, int ni, char *argv[], int P, double del_x, double del_y, double del_t, double Acnst, double Bcnst, double kappaC, double kappaPhi, double matcomp, double partcomp, double *vtotX_d, double *vtotY_d, 
double *velX_d, double *velY_d, double emob, double *sum_phi_d, double *sumsq_phi_d, double *sumcub_phi_d, double Dvol, double Dvap, double Dsurf, double Dgb, double gsize, int op_steps, int num_steps, int threadsPerBlock_X, int threadsPerBlock_Y, int nprocs, int top, int bottom, int Nx, int Ny, int rank);

void Init_Conf(double *phi, double *comp, double del_x, int nx, int ny,int ni, int Nx, int Ny, int rank,double matcomp, double gsize, double partcomp );

void Output_Conf(double *phi, double *comp, int nx, int ny,int ni, int P, char *argv[], int rank);

int main(int argc, char * argv[]) {

  FILE *fw, *fr;
  char NAME[950], hostname[HOST_NAME_MAX + 1], string1[1000];
  
  int i, j, k, l;
  int P;  // �ð� �ܰ� ����
  // Time step variable

  double *phi, *comp;  // ���� �迭�� ���� �迭
  // Material and composition arrays
  
  long SEED;  // ���� ���� �õ�
  // Seed for random number generation
  
  int blocks, blocks_xy;  // CUDA ���� �� XY ���� ��
  // CUDA block and XY block counts
  int threadsPerBlock_X, threadsPerBlock_Y;  // CUDA ������ �� (X, Y)
  // CUDA thread counts (X, Y)
  int num_steps, count, op_steps;  // �ùķ��̼� �ܰ�, ���?ī��Ʈ, ���?�ܰ�
  // Simulation steps, calculation count, output steps
  double del_x, del_y, del_t;  // ���� �� �ð� ����
  // Spatial and temporal intervals
  double sim_time, total_time;  // �ùķ��̼� �ð� �� �� �ð�
  // Simulation time and total time
  int Nx, Ny;  // ��ü ������ ũ�� (X, Y)
  // Overall domain size (X, Y)
  int nx, ny, ni, np;  // ���� ������ ũ�� (X, Y), ��Ÿ ����
  // Local domain size (X, Y), other variables
  
  double *phi_old_d, *phi_new_d;  // ���� �� ���ο� ���� ���� ������
  // Pointers to old and new state variables
  int *eta_d, *num_phi_d, *zeta_d;  // CUDA ���� ������
  // CUDA variable pointers
  double *comp_d, *comp_new_d, *uc_d, *mu_d, *fen_d, *dummy_d;
  double *sum_phi_d, *sumsq_phi_d, *sumcub_phi_d;
  double *vtotX_d, *vtotY_d;
  double *velX_d, *velY_d;
  double *sum_uc_d, *sum_fen_d, *lamda_d;
  double *dphi_d;
  double *Diff, *fenergy, *Ceq;
  double *diff_d;
  double *Dvol_d, *Dvap_d, *Dsur_d, *Dgbd_d;
  int *eta;
  double emob, emtj, sigma, delta, tifac, garea, tot_area,gsize;
  double Dvol, Dvap, Dsurf, Dgb;

  double *dforX_d, *dforY_d, *dtor_d;
  double *forcX_d, *forcY_d, *torq_d, *radcX_d, *radcY_d, *vol_d;
  double kappaRbm, phicoff, compcoff, Mtra, Mrot;
  
  double matcomp, partcomp;  // ���� �� ���� ����
  // Material and particle composition
  double Acnst, Bcnst, kappaC, kappaPhi;  // ���?�� Ŀ�н��Ͻ�
  // Constants and capacitance
  int adv_flag, device_flag;  // �߰� �� ��ġ �÷���
  // Advance and device flags
  int res_stat, res_steps;  // �����?���� �� �ܰ�
  // Restart status and steps

  int rank, top, bottom, nprocs;  // MPI ��ũ, ����/���� �̿�, ���μ��� ��
  // MPI rank, top/bottom neighbors, number of processes
  int tag_0 = 10; 
  int tag_n = 20;
  MPI_Request sreq_0, sreq_n;  // �񵿱� MPI ���?��û
  // Asynchronous MPI communication requests
  MPI_Status status_0, status_n;  // MPI ���� ����
  // MPI status variables

  time_t tim;  // ���� �ð��� �����ϴ� ����
  // Variable to store the current time
  
  // MPI �ʱ�ȭ
  // Initialize MPI
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);  // ���μ��� �� ���?  // Get the number of processes
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);  // ���� ���μ����� ��ũ ���?  // Get the rank of the current process
  if ( rank == 0 ) printf("No. of processors in execution %d\n", nprocs);
  
  // ���� ȣ��Ʈ �̸� ��������
  // Get the hostname of the current process
  gethostname(hostname, HOST_NAME_MAX + 1);
  printf("Host name: %s for rank = %d\n\n", hostname, rank);

  // �Է� ���� ����
  // Open input file
  fr = fopen(argv[1],"r");
  if(fr == NULL) {  // ������ ���� ���� ���?���� �޽��� ���?    // If the file cannot be opened, print an error message
    printf("ERROR: No input file \n \t or \n");
    printf("ERROR: file %s not found\n", argv[1]);
    printf("\nRunning Syntax:\n \t \tmpirun -np <No_of_Processors> ./pfm_mpi_cuda.out <input_file> \n");
    printf("\nExiting\n");
    exit(2);
  }

  // �Է� ���Ͽ��� �Ű� ���� �б�
  // Read parameters from input file
  fscanf(fr, "%s %d\n",  string1, &Nx);
  fscanf(fr, "%s %d\n",  string1, &Ny);
  fscanf(fr, "%s %d\n",  string1, &ni);
  fscanf(fr, "%s %le\n", string1, &del_x);
  fscanf(fr, "%s %le\n", string1, &del_y);
  fscanf(fr, "%s %le\n", string1, &del_t);
  fscanf(fr, "%s %d\n", string1, &blocks);
  fscanf(fr, "%s %d\n",  string1, &threadsPerBlock_X);
  fscanf(fr, "%s %d\n",  string1, &threadsPerBlock_Y);
  fscanf(fr, "%s %ld\n",  string1, &SEED);
  fscanf(fr, "%s %d\n",  string1, &op_steps);
  fscanf(fr, "%s %d\n",  string1, &num_steps);
  fscanf(fr, "%s %lf\n", string1, &matcomp);
  fscanf(fr, "%s %lf\n", string1, &partcomp);
  fscanf(fr, "%s %le\n", string1, &Acnst);
  fscanf(fr, "%s %le\n", string1, &Bcnst);
  fscanf(fr, "%s %le\n", string1, &kappaC);
  fscanf(fr, "%s %le\n", string1, &kappaPhi);
  fscanf(fr, "%s %le\n", string1, &emob);
  fscanf(fr, "%s %le\n", string1, &Dvol);
  fscanf(fr, "%s %le\n", string1, &Dvap);
  fscanf(fr, "%s %le\n", string1, &Dsurf);
  fscanf(fr, "%s %le\n", string1, &Dgb);
  fscanf(fr, "%s %lf\n", string1, &gsize);
  fscanf(fr, "%s %lf\n", string1, &kappaRbm);
  fscanf(fr, "%s %lf\n", string1, &phicoff);
  fscanf(fr, "%s %lf\n", string1, &compcoff);
  fscanf(fr, "%s %lf\n", string1, &Mtra);
  fscanf(fr, "%s %lf\n", string1, &Mrot);
  fscanf(fr, "%s %d\n", string1, &res_stat);
  fscanf(fr, "%s %d\n", string1, &res_steps);
  fscanf(fr, "%s %d\n", string1, &adv_flag);
  fscanf(fr, "%s %d\n", string1, &device_flag);
  fclose(fr);
  
  // �� ���μ����� ������ ũ�� ���?  // Calculate the domain size for each process
  nx = Nx / nprocs + 2; 

  // Nx�� ���μ��� ���� ���������� �ʴ� ���?���� �޽��� ���?  // Print an error message if Nx is not divisible by the number of processes
  if ( Nx % nprocs != 0 ) { 
    printf("ERROR: Nx is not divisible by no. of processors (np). Choose them accordingly.\n");
    exit(1);
    printf("Exiting\n");
  }
  ny = Ny;
  
  // �����?�Ű� ������ ���Ͽ� ����
  // Save the used parameters to a file
  if ( rank == 0 ) {
    sprintf(NAME,"paramters_used_For_%s", argv[1]);
    fw = fopen(NAME,"w");
    time(&tim);
    fprintf(fw, "Name of input file: %s\n\n", argv[1]);
    fprintf(fw, "Simulation date and time: %s\n", ctime(&tim));
    fprintf(fw, "Nx = %d\n", nx);
    fprintf(fw, "Ny = %d\n", ny);
    fprintf(fw, "ni = %d\n", ni);
    fprintf(fw, "del_x = %le\n", del_x);
    fprintf(fw, "del_y = %le\n", del_y);
    fprintf(fw, "del_t = %le\n", del_t);
    fprintf(fw, "Block size = %d\n", blocks);
    fprintf(fw, "Threads per block (x)  = %d\n", threadsPerBlock_X);
    fprintf(fw, "Threads per block (y)  = %d\n", threadsPerBlock_Y);
    fprintf(fw, "Seed = %ld\n", SEED);
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
  
  MPI_Barrier(MPI_COMM_WORLD);
  
  // �� ���μ����� ���� �� ���� �̿� ���μ��� ����
  // Set top and bottom neighboring processes for each process
  top = rank - 1; 
  if ( rank == 0 ) { 
    top = nprocs - 1;
  }
  bottom = rank + 1; 
  if ( rank == nprocs-1 ) { 
    bottom = 0;
  }
  printf("( top, rank, bottom ) = ( %d, %d, %d )\n", top, rank, bottom);

  // ���� �� ���� �迭 �Ҵ�
  // Allocate composition and material arrays
  comp = (double *)malloc(sizeof (double)*nx*ny);
  phi  = (double *)malloc(sizeof (double)*nx*ny*ni);
  MPI_Barrier(MPI_COMM_WORLD);
  
  // �ʱ� ������ ����
  // Set up initial domain
  if ( rank == 0 ) printf("\nSetting up initial domain:\n");
  Init_Conf(phi, comp, del_x, nx, ny,ni, Nx, Ny, rank,matcomp, gsize, partcomp );
  MPI_Barrier(MPI_COMM_WORLD);

  // ���?������ ��ȯ
  // Exchange boundary data
  MPI_Isend(&phi[0 + ni * (0 + (1)*ny)],    ni*ny, MPI_DOUBLE,    top, tag_0, MPI_COMM_WORLD, &sreq_0);
  MPI_Recv (&phi[0 + ni * (0 + (nx-1)*ny)], ni*ny, MPI_DOUBLE, bottom, tag_0, MPI_COMM_WORLD, &status_0);
  MPI_Wait(&sreq_0, &status_0);

  MPI_Isend(&phi[0 + ni * (0 + (nx-2)*ny)], ni*ny, MPI_DOUBLE, bottom, tag_n, MPI_COMM_WORLD, &sreq_n);
  MPI_Recv (&phi[0],             ni*ny, MPI_DOUBLE,    top, tag_n, MPI_COMM_WORLD, &status_n);
  MPI_Wait(&sreq_n, &status_n);

  MPI_Barrier(MPI_COMM_WORLD);

  // ���� ������ ��ȯ
  // Exchange composition data
  MPI_Isend(&comp[0 + (1)*ny],    ny, MPI_DOUBLE,    top, tag_0, MPI_COMM_WORLD, &sreq_0);
  MPI_Recv (&comp[0 + (nx-1)*ny], ny, MPI_DOUBLE, bottom, tag_0, MPI_COMM_WORLD, &status_0);
  MPI_Wait(&sreq_0, &status_0);

  MPI_Isend(&comp[0 + (nx-2)*ny], ny, MPI_DOUBLE, bottom, tag_n, MPI_COMM_WORLD, &sreq_n);
  MPI_Recv (&comp[0],             ny, MPI_DOUBLE,    top, tag_n, MPI_COMM_WORLD, &status_n);
  MPI_Wait(&sreq_n, &status_n);

  MPI_Barrier(MPI_COMM_WORLD);

  // �ʱ� ������ ����
  // Save initial domain
  if ( rank == 0 ) printf("Writing initial domain\n");
  P=0;
  Output_Conf(phi, comp, nx, ny, ni, P, argv, rank);
  MPI_Barrier(MPI_COMM_WORLD);
  
  // GPU �Լ� ȣ��
  // Call GPU function
  pfm_mpi_gpu(phi, comp, nx, ny, np, ni, argv, P, del_x, del_y, del_t, Acnst, Bcnst, kappaC, kappaPhi, matcomp, partcomp, vtotX_d, vtotY_d,velX_d, velY_d, emob, sum_phi_d, sumsq_phi_d, sumcub_phi_d,Dvol, Dvap, Dsurf, Dgb, gsize,op_steps, num_steps, 
  threadsPerBlock_X, threadsPerBlock_Y, nprocs, top, bottom, Nx, Ny, rank);

  MPI_Barrier(MPI_COMM_WORLD);

  // �޸� ����
  // Free memory
  free(phi);
  free(comp);
  
  if ( rank == 0 ) printf("\nCode execution has completed\n");
  if ( rank == 0 ) printf("\nDone; time to say good-bye!\n\n");

  // MPI ����
  // Finalize MPI
  MPI_Finalize();

}

