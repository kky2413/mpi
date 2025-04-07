void Init_Conf(double *phi, int *gid, double *comp, int *nap_phi, int cmpres, int mixrat, 
  double del_x, int nx, int ny, int nz, int ni, int np,int Nx, int Ny, int Nz, int rank, 
  double matcomp, double partcomp, int mx, int my, int mz){
 double rmin;
 int i, j, h, k, l, ig;
 FILE *read;
 char fdat[30]; 
 int iii, jjj, kkk, hhh;
 int i33, j33, k33;
 double *psi = (double *)malloc(Nx*Ny*Nz*ni* sizeof(double)); 
 int *gsd = (int *)malloc(Nx*Ny*Nz*ni* sizeof(int));
 double *csm = (double *)malloc(Nx*Ny*Nz* sizeof(double));
 int *nsp_phi = (int *)malloc(Nx*Ny*Nz* sizeof(int));
 if (csm == NULL) {
  fprintf(stderr, "Memory allocation failed!------------csm\n");
 }

        for (i = 0; i < Nx; i++) {
         for (j = 0; j < Ny; j++) {
          for (h = 0; h < Nz; h++) {
           for (k = 0; k < ni; k++) {
            psi[k + (h + (j + i * Ny) * Nz) * ni] = 0.0;
            gsd[k + (h + (j + i * Ny) * Nz) * ni] = -1;
           }
           csm[h + (j + i * Ny) * Nz] = matcomp;
           nsp_phi[h + (j + i * Ny) * Nz] = 0;
          }
         }
        }
        
        sprintf(fdat,"../input/cp%04d_%04d.txt", cmpres, mixrat);
        read = fopen (fdat, "r");
        if (read == NULL) {
          fprintf(stderr, "Error: Unable to open input file \n");
        }
      
          for (i = 0; i < mx; i++){
              for (j = 0; j < my; j++){
               for (k = 0; k < mz; k++){
      
                  fscanf(read,"%d %d %d %lf %d", &iii, &jjj, &kkk, &rmin, &hhh);
      
                  if (hhh > 0 && kkk < Nz) {
                          i33 = iii;
                          j33 = jjj;
                          k33 = kkk;
                          if (hhh > np || hhh < 1) printf("Error:%d", hhh);
                        psi[0 + (k33 + (j33 + i33 * Ny) * Nz) * ni] = 1.0;
                        gsd[0 + (k33 + (j33 + i33 * Ny) * Nz) * ni] = hhh - 1;
                        csm[k33 + (j33 + i33 * Ny) * Nz] = partcomp;
                        nsp_phi[k33 + (j33 + i33 * Ny) * Nz] = 1;
                  }
                }
              }
          }

        fclose(read);

        for (i = 1; i < nx-1; i++) {
          for (j = 0; j < ny; j++) {
           for (h = 0; h < nz; h++) {
            ig = i - 1 + rank * (nx - 2);
            for (k = 0; k < ni; k++) {
            phi[k + (h + (j + i * ny) * nz) * ni] = psi[k + (h + (j + ig * Ny) * Nz) * ni];
            gid[k + (h + (j + i * ny) * nz) * ni] = gsd[k + (h + (j + ig * Ny) * Nz) * ni];
            }
            comp[h + (j + i * ny) * nz] = csm[h + (j + ig * Ny) * Nz];
            nap_phi[h + (j + i * ny) * nz] = nsp_phi[h + (j + ig * Ny) * Nz];
           }
          }
         }

  free(psi);
  free(gsd);
  free(csm);
  free(nsp_phi);
}
