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

#include "savetimestep.cu"  
#include "Initialize_domain.cu"

void pfm_mpi_gpu(double *phi, int *gid, double *comp, int *nap_phi, int nx, int ny, int nz, int np, int ni, double alpha, double beta,char *argv[], int P, 
  double del_x, double del_y, double del_t, double del_z, double Acnst, double Bcnst, double kappaC, double kappaPhi, double matcomp, double partcomp, 
  double emob, double *sum_phi_d, double *sumsq_phi_d, double *sumcub_phi_d, double Dvol, double Dvap, double Dsurf, double Dgb, double gsize, int op_steps, 
  int num_steps, int threadsPerBlock_X, int threadsPerBlock_Y, int nprocs, int top, int bottom, int Nx, int Ny, int Nz,int rank, long SEED, int cmpres, int mixrat, 
  double kappaRbm, double phicoff, double compcoff, double Mtra, double Mrot,   double pss, double www, double cep, double emm);

void Init_Conf(double *phi, int *gid, double *comp, int *nap_phi, int cmpres, int mixrat, 
  double del_x, int nx, int ny, int nz, int ni, int np,int Nx, int Ny, int Nz, int rank, 
  double matcomp, double partcomp, int mx, int my, int mz);

void Output_Conf(double *phi, int *gid, double *comp, int *nap_phi, int nx, int ny, int nz, int ni,int P, int rank, int nprocs, char *argv[], int Nx);

int main(int argc, char * argv[]) {

  FILE *fpin, *fpcout;
  char finput[25] = "src/TernarySolve";
  char NAME[950], hostname[HOST_NAME_MAX + 1], param[1000];
  
  int i, j, k, l;
  int P; 

  double *phi, *comp;  
  
  double rmin;

  long SEED; 
  
  int blocks, blocks_xy;  
  int threadsPerBlock_X, threadsPerBlock_Y; 
  int num_steps, count, op_steps; 
  double del_x, del_y, del_z, del_t;
  int mx, my, mz, Mx;
  double sim_time, total_time;
  int Nx, Ny, Nz;
  int nx, ny, ni, nz, np,dnum = 3;  
  
  double *phi_old_d, *phi_new_d; 
  int *eta_d, *num_phi_d, *zeta_d;  
  double *comp_d, *comp_new_d, *uc_d, *mu_d, *fen_d, *dummy_d;
  double *sum_phi_d, *sumsq_phi_d, *sumcub_phi_d;
  double *sum_uc_d, *sum_fen_d, *lamda_d;
  double *dphi_d;
  double *Diff, *fenergy, *Ceq;
  double *diff_d;
  double *Dvol_d, *Dvap_d, *Dsur_d, *Dgbd_d;
  double emob, emtj, sigma, delta, tifac, garea, tot_area,gsize;
  double Dvol, Dvap, Dsurf, Dgb;
  double tf, alpha, beta, Dmax;
  int cmpres, mixrat;

  double kappaRbm, phicoff, compcoff, Mtra, Mrot;
  double pss, www, cep, emm;
  
  double matcomp, partcomp;
  int *gid, *nap_phi;
  int *gid_d, *nap_phi_d;  
  double Acnst, Bcnst, kappaC, kappaPhi;  
  int adv_flag, device_flag; 
  int res_stat, res_steps;  

  int rank, top, bottom, nprocs; 
  int tag_0 = 10; 
  int tag_n = 20;
  MPI_Request sreq_0, sreq_n; 
  MPI_Status status_0, status_n;

  time_t tim;
  
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs); 
  MPI_Comm_rank(MPI_COMM_WORLD, &rank); 
  if ( rank == 0 ) printf("No. of processors in execution %d\n", nprocs);
  
  gethostname(hostname, HOST_NAME_MAX + 1);
  printf("Host name: %s for rank = %d\n\n", hostname, rank);

  fpin = fopen(argv[1],"r");
  if(fpin == NULL) { 
    printf("ERROR: No input file \n \t or \n");
    printf("ERROR: file %s not found\n", argv[1]);
    printf("\nRunning Syntax:\n \t \tmpirun -np <No_of_Processors> ./pfm_mpi_cuda.out <input_file> \n");
    printf("\nExiting\n");
    exit(2);
  }

  if(fscanf (fpin, "%s%d", param,&mx));
  if(fscanf (fpin, "%s%d", param,&my));
  if(fscanf (fpin, "%s%d", param,&mz));
  if(fscanf (fpin, "%s%d", param,&np));
  if(fscanf (fpin, "%s%d", param,&ni));
  if(fscanf (fpin, "%s%lf",param,&del_x));
  if(fscanf (fpin, "%s%lf",param,&del_y));
  if(fscanf (fpin, "%s%lf",param,&del_z));
  if(fscanf (fpin, "%s%le",param,&tf));
  if(fscanf (fpin, "%s%d", param,&cmpres));
  if(fscanf (fpin, "%s%d", param,&mixrat));
  if(fscanf (fpin, "%s%ld",param,&SEED));
  if(fscanf (fpin, "%s%d", param,&op_steps));
  if(fscanf (fpin, "%s%d", param,&num_steps));
  if(fscanf (fpin, "%s%lf",param,&matcomp));
  if(fscanf (fpin, "%s%lf",param,&partcomp));
  if(fscanf (fpin, "%s%lf",param,&Acnst));
  if(fscanf (fpin, "%s%lf",param,&Bcnst));
  if(fscanf (fpin, "%s%lf",param,&kappaC));
  if(fscanf (fpin, "%s%lf",param,&kappaPhi));
  if(fscanf (fpin, "%s%le",param,&emob));
  if(fscanf (fpin, "%s%le",param,&Dvol));
  if(fscanf (fpin, "%s%le",param,&Dvap));
  if(fscanf (fpin, "%s%le",param,&Dsurf));
  if(fscanf (fpin, "%s%le",param,&Dgb));
  if(fscanf (fpin, "%s%lf",param,&gsize));
  if(fscanf (fpin, "%s%lf",param,&pss));
  if(fscanf (fpin, "%s%lf",param,&kappaRbm));
  if(fscanf (fpin, "%s%lf",param,&phicoff));
  if(fscanf (fpin, "%s%lf",param,&compcoff));
  if(fscanf (fpin, "%s%lf",param,&Mtra));
  if(fscanf (fpin, "%s%lf",param,&Mrot));
  if(fscanf (fpin, "%s%d", param,&res_stat));
  if(fscanf (fpin, "%s%d", param,&res_steps));
  if(fscanf (fpin, "%s%d", param,&adv_flag));
  if(fscanf (fpin, "%s%d", param,&device_flag));

  fclose(fpin);

  Dmax = Dsurf;
  if (Dvol > Dmax) Dmax = Dvol;
  if (Dvap > Dmax) Dmax = Dvap;
  if (Dgb  > Dmax) Dmax = Dgb;
  alpha = kappaC * Dmax/4.0;
  beta  = (Acnst + Bcnst) * Dmax/4.0;
  del_t = tf * del_x * del_x * del_x * del_x / ((16.0 * alpha) * 3);

  if (mx%8 == 0) Nx = mx; else Nx = (mx/8 + 1) * 8;
  if (my%8 == 0) Ny = my; else Ny = (my/8 + 1) * 8;
  if (mz%8 == 0) Nz = mz; else Nz = (mz/8 + 1) * 8;

  nx = Nx/nprocs + 2; 

  if ( Nx % nprocs != 0 ) { 
    printf("ERROR: Nx is not divisible by no. of processors (np). Choose them accordingly.\n");
    exit(1);
    printf("Exiting\n");
  }
  ny = Ny;
  nz = Nz;
  
  if ( rank == 0 ) {
    sprintf(NAME,"paramters_used_For_%s", argv[1]);
    fpcout = fopen(NAME,"w");
    fprintf(fpcout, "mx = %d\n", mx);
    fprintf(fpcout, "my = %d\n", my);
    fprintf(fpcout, "mz = %d\n", mz);
    fprintf(fpcout, "nx = %d\n", nx);
    fprintf(fpcout, "ny = %d\n", ny);
    fprintf(fpcout, "nz = %d\n", nz);
    fprintf(fpcout, "Nx = %d\n", Nx);
    fprintf(fpcout, "Ny = %d\n", Ny);
    fprintf(fpcout, "Nz = %d\n", Nz);
    fprintf(fpcout, "np = %d\n", np);
    fprintf(fpcout, "ni = %d\n", ni);
    fprintf(fpcout, "del_x = %lf\n", del_x);
    fprintf(fpcout, "del_y = %lf\n", del_y);
    fprintf(fpcout, "del_z = %lf\n", del_z);
    fprintf(fpcout, "del_t = %lf\n", del_t);
    fprintf(fpcout, "tf = %le\n", tf);
    fprintf(fpcout, "cmpres = %d\n", cmpres);
    fprintf(fpcout, "mixrat = %d\n", mixrat);
    fprintf(fpcout, "SEED = %ld\n", SEED);
    fprintf(fpcout, "op_steps = %d\n", op_steps);
    fprintf(fpcout, "num_steps = %d\n", num_steps);
    fprintf(fpcout, "matcomp = %lf\n", matcomp);
    fprintf(fpcout, "partcomp = %lf\n", partcomp);
    fprintf(fpcout, "Acnst = %lf\n", Acnst);
    fprintf(fpcout, "Bcnst = %lf\n", Bcnst);
    fprintf(fpcout, "kappaC = %lf\n", kappaC);
    fprintf(fpcout, "kappaPhi = %lf\n", kappaPhi);
    fprintf(fpcout, "emob = %le\n", emob);
    fprintf(fpcout, "Dvol = %le\n", Dvol);
    fprintf(fpcout, "Dvap = %le\n", Dvap);
    fprintf(fpcout, "Dsurf = %le\n", Dsurf);
    fprintf(fpcout, "Dgb = %le\n", Dgb);
    fprintf(fpcout, "gsize = %lf\n", gsize);
    fprintf(fpcout, "pss = %lf\n", pss);
    fprintf(fpcout, "kappaRbm = %lf\n", kappaRbm);
    fprintf(fpcout, "phicoff = %lf\n", phicoff);
    fprintf(fpcout, "compcoff = %lf\n", compcoff);
    fprintf(fpcout, "Mtra = %lf\n", Mtra);
    fprintf(fpcout, "Mrot = %lf\n", Mrot);
    fprintf(fpcout, "res_stat = %d\n", res_stat);
    fprintf(fpcout, "res_steps = %d\n", res_steps);
    fprintf(fpcout, "adv_flag = %d\n", adv_flag);
    fprintf(fpcout, "device_flag = %d\n", device_flag);

    fclose (fpcout);
  }
  MPI_Barrier(MPI_COMM_WORLD);
  
  top = rank - 1; 
  if ( rank == 0 ) { 
    top = nprocs - 1;
  }
  bottom = rank + 1; 
  if ( rank == nprocs-1 ) { 
    bottom = 0;
  }
  printf("( top, rank, bottom ) = ( %d, %d, %d )\n", top, rank, bottom);

  comp = (double *)malloc(sizeof (double) * nx * ny * nz);
  phi = (double *)malloc(sizeof (double) * nx * ny * nz * ni);
  nap_phi = (int *)malloc(sizeof (int) * nx * ny * nz);
  gid = (int *)malloc(sizeof (int) * nx * ny * nz * ni);
  MPI_Barrier(MPI_COMM_WORLD);
  
  if ( rank == 0 ) printf("\nSetting up initial domain:\n");
  Init_Conf(phi, gid, comp, nap_phi, cmpres, mixrat, del_x, nx, ny, nz, ni, np, Nx, Ny, Nz, rank, matcomp, partcomp, mx, my, mz);
  MPI_Barrier(MPI_COMM_WORLD);

  MPI_Isend(&phi[0 + (0 + nz * (0 + (1) * ny)) * ni],    ni*ny*nz, MPI_DOUBLE,    top, tag_0, MPI_COMM_WORLD, &sreq_0);
  MPI_Recv (&phi[0 + (0 + nz * (0 + (nx-1) * ny)) * ni], ni*ny*nz, MPI_DOUBLE, bottom, tag_0, MPI_COMM_WORLD, &status_0);
  MPI_Wait(&sreq_0, &status_0);

  MPI_Isend(&phi[0 + (0 + nz * (0 + (nx-2) * ny)) * ni], ni*ny*nz, MPI_DOUBLE, bottom, tag_n, MPI_COMM_WORLD, &sreq_n);
  MPI_Recv (&phi[0],             ni*ny*nz, MPI_DOUBLE,    top, tag_n, MPI_COMM_WORLD, &status_n);
  MPI_Wait(&sreq_n, &status_n);

  MPI_Barrier(MPI_COMM_WORLD);

  MPI_Isend(&gid[0 + (0 + nz * (0 + (1) * ny)) * ni],    ni*ny*nz, MPI_INT,    top, tag_0, MPI_COMM_WORLD, &sreq_0);
  MPI_Recv (&gid[0 + (0 + nz * (0 + (nx-1) * ny)) * ni], ni*ny*nz, MPI_INT, bottom, tag_0, MPI_COMM_WORLD, &status_0);
  MPI_Wait(&sreq_0, &status_0);

  MPI_Isend(&gid[0 + (0 + nz * (0 + (nx-2) * ny)) * ni], ni*ny*nz, MPI_INT, bottom, tag_n, MPI_COMM_WORLD, &sreq_n);
  MPI_Recv (&gid[0],             ni*ny*nz, MPI_INT,    top, tag_n, MPI_COMM_WORLD, &status_n);
  MPI_Wait(&sreq_n, &status_n);

  MPI_Barrier(MPI_COMM_WORLD);

  MPI_Isend(&comp[0 + nz * (0 + (1)*ny)],    ny*nz, MPI_DOUBLE,    top, tag_0, MPI_COMM_WORLD, &sreq_0);
  MPI_Recv (&comp[0 + nz * (0 + (nx-1)*ny)], ny*nz, MPI_DOUBLE, bottom, tag_0, MPI_COMM_WORLD, &status_0);
  MPI_Wait(&sreq_0, &status_0);

  MPI_Isend(&comp[0 + nz * (0 + (nx-2)*ny)], ny*nz, MPI_DOUBLE, bottom, tag_n, MPI_COMM_WORLD, &sreq_n);
  MPI_Recv (&comp[0],             ny*nz, MPI_DOUBLE,    top, tag_n, MPI_COMM_WORLD, &status_n);
  MPI_Wait(&sreq_n, &status_n);

  MPI_Barrier(MPI_COMM_WORLD);

  MPI_Isend(&nap_phi[0 + nz * (0 + (1)*ny)],    ny*nz, MPI_INT,    top, tag_0, MPI_COMM_WORLD, &sreq_0);
  MPI_Recv (&nap_phi[0 + nz * (0 + (nx-1)*ny)], ny*nz, MPI_INT, bottom, tag_0, MPI_COMM_WORLD, &status_0);
  MPI_Wait(&sreq_0, &status_0);

  MPI_Isend(&nap_phi[0 + nz * (0 + (nx-2)*ny)], ny*nz, MPI_INT, bottom, tag_n, MPI_COMM_WORLD, &sreq_n);
  MPI_Recv (&nap_phi[0],             ny*nz, MPI_INT,    top, tag_n, MPI_COMM_WORLD, &status_n);
  MPI_Wait(&sreq_n, &status_n);

  MPI_Barrier(MPI_COMM_WORLD);

  if ( rank == 0 ) printf("Writing initial domain\n");
  P=0;
  Output_Conf(phi, gid, comp, nap_phi, nx, ny, nz, ni, P, rank, nprocs, argv, Nx);
  MPI_Barrier(MPI_COMM_WORLD);
  
  pfm_mpi_gpu(phi, gid, comp, nap_phi, nx, ny, nz, np, ni, alpha, beta, argv, P, del_x, del_y, del_t, del_z, Acnst, Bcnst, kappaC, kappaPhi, matcomp, partcomp, emob, sum_phi_d, sumsq_phi_d, 
    sumcub_phi_d, Dvol, Dvap, Dsurf, Dgb, gsize, op_steps, num_steps, threadsPerBlock_X, threadsPerBlock_Y, nprocs, top, bottom, Nx, Ny, Nz, rank, SEED, cmpres, mixrat, kappaRbm, phicoff, 
    compcoff, Mtra, Mrot, pss, www, cep, emm);

  MPI_Barrier(MPI_COMM_WORLD);

  free(nap_phi);
  free(gid);
  free(phi);
  free(comp);
  
  if ( rank == 0 ) printf("\nCode execution has completed\n");
  if ( rank == 0 ) printf("\nDone; time to say good-bye!\n\n");

  MPI_Finalize();

}
