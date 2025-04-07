void Output_Conf(double *phi, int *gid, double *comp, int *nap_phi, int nx, int ny, int nz, int ni,int P, int rank, int nprocs, char *argv[], int Nx) {
    FILE  *fp1, *fp2, *fp3;
    char  fname1[100], fname2[100], fname3[100];
    int i, j, h, k;
    double phimax, phival, nmax;
    int index;  

    sprintf(fname1, "../output/comp_data/comp%09d_%d.dat", P, rank);
    sprintf(fname2, "../output/phi_data/phi%09d_%d.dat", P, rank);
    sprintf(fname3, "../output/nmax_data/nmax%09d_%d.dat", P, rank);
    
    fp1 = fopen(fname1, "w");
    fp2 = fopen(fname2, "w");
    fp3 = fopen(fname3, "w");
    
    for (i = 1; i < nx-1; i++) {
        for (j = 0; j < ny; j++){
            for (h = 0; h < nz; h++){
                int index_1 = h + (j + i * ny) * nz;    
                phimax = 0.0;
                nmax = 0.0;
                index = nap_phi[h + (j + i * ny) * nz];
                for (k = 0; k < index; k++){
                    if (phi[k + (h + (j + i * ny) * nz) * ni] > phimax) {
                        phimax = phi[k + (h + (j + i * ny) * nz) * ni];
                        nmax   = (double)gid[k + (h + (j + i * ny) * nz) * ni];
                    }
                }
                if (phimax < 1.0e-8) nmax = -1.0;
                fprintf(fp1, "%lf\n", comp[index_1]);
                fprintf(fp2, "%lf\n", phimax);
                fprintf(fp3, "%lf\n",nmax);
            }
        }
    }
    fclose(fp1);
    fclose(fp2);
    fclose(fp3);

}



