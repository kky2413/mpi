void Init_Conf(double *phi, double *comp, double del_x, int nx, int ny,int ni, int Nx, int Ny, int rank,double matcomp, double gsize, double partcomp ) {

    int i, j,k, index, ig;
    double *centx, *centy;
    int imax, jmax, imin, jmin; 
    double *psi = (double *)malloc(Nx * Ny * ni * sizeof(double)); 
    double *cop = (double *)malloc(Nx * Ny * sizeof(double));
    centx = (double*)malloc(sizeof(double) * ni);
    centy = (double*)malloc(sizeof(double) * ni);
    
        for (i = 0; i < Nx; i++) {
         for (j = 0; j < Ny; j++) {
            cop[j + i * Ny] = matcomp;
          for (k = 0; k < ni; k++) {
            psi[k + (j + i * Ny) * ni] = 0.0;
            }
        }
    }
    
            if (ni == 2) {
                centx[0] = (double)(Nx * 0.5 - gsize);
                centy[0] = (double)Ny * 0.5;
                centy[1] = (double)Ny * 0.5;
                centx[1] = (double)(Nx * 0.5 + gsize);
            }

            if (ni == 8) {
                centx[5] = (double)Nx * 0.5 - gsize;
                centy[5] = (double)Nx * 0.5 - gsize;

                centx[0] = centx[5] - (2.0 * gsize + 1.0);
                centy[0] = centy[5] - (2.0 * gsize + 1.0);

                centx[6] = centx[5] + (2.0 * gsize + 1.0);
                centx[7] = centx[6] + (2.0 * gsize + 1.0);

                centx[4] = centx[0];
                centx[1] = centx[5];
                centx[2] = centx[6];
                centx[3] = centx[7];

                centy[1] = centy[0];
                centy[2] = centy[0];
                centy[3] = centy[0];

                centy[4] = centy[5];
                centy[6] = centy[5];
                centy[7] = centy[5];
            }

            if (ni == 16) {
                centx[5] = (double)Nx * 0.5 - gsize;
                centy[5] = (double)Ny * 0.5 - gsize;

                centx[0] = centx[5] - (2.0 * gsize + 1.0);
                centy[0] = centy[5] - (2.0 * gsize + 1.0);

                centx[10] = centx[5] + (2.0 * gsize + 1.0);
                centy[10] = centy[5] + (2.0 * gsize + 1.0);

                centx[15] = centx[10] + (2.0 * gsize + 1.0);
                centy[15] = centy[10] + (2.0 * gsize + 1.0);

                centx[4] = centx[0];
                centx[8] = centx[0];
                centx[12] = centx[0];

                centx[1] = centx[5];
                centx[9] = centx[5];
                centx[13] = centx[5];

                centx[2] = centx[10];
                centx[6] = centx[10];
                centx[14] = centx[10];

                centx[3] = centx[15];
                centx[7] = centx[15];
                centx[11] = centx[15];

                centy[1] = centy[0];
                centy[2] = centy[0];
                centy[3] = centy[0];

                centy[4] = centy[5];
                centy[6] = centy[5];
                centy[7] = centy[5];

                centy[8] = centy[10];
                centy[9] = centy[10];
                centy[11] = centy[10];

                centy[12] = centy[15];
                centy[13] = centy[15];
                centy[14] = centy[15];
            }

    for (k = 0; k < ni; k++) {
        imin = (int)(centx[k] - 1.2 * gsize);
        imax = (int)(centx[k] + 1.2 * gsize);
        jmin = (int)(centy[k] - 1.2 * gsize);
        jmax = (int)(centy[k] + 1.2 * gsize);

  for (i = imin; i < imax; i++) {
          for (j = jmin; j < jmax; j++) {
	    if (sqrt(((double)i - centx[k])*((double)i - centx[k]) + ((double)j - centy[k])*((double)j - centy[k])) <= gsize) {
             cop[j + i * Ny] = partcomp;
             psi[k + (j + i * Ny) * ni] = 1.0;

                }
            }
        }
    }
  for (i = 1; i < nx - 1; i++) {
        for (j = 0; j < ny; j++) {
         index = j + i * ny;
         ig = i - 1 + rank * (nx - 2);  // 글로벌 인덱스 ig를 계산
            for (k = 0; k < ni; k++) {
            phi[k + (index) * ni] = psi[k + (j + ig * Ny) * ni];
          }
          comp[index] = cop[j + ig * Ny];
        }
    }

    free(centx);
    free(centy);
    free(psi);
    free(cop);
}
