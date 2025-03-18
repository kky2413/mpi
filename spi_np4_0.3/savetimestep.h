//output 파일들을 생성하는 코드
//making output files
void savetimestep(double *phi, int nx, int ny, int P, int rank, char *argv[]) { 

  FILE *fw;
  int i, j, index;
  char fname[950];

  sprintf(fname,"result/prof_%s_%08d_%d.dat", argv[1], P, rank);
  fw = fopen(fname,"w");
  for (i = 0; i < nx; i++) {
    for (j = 0; j < ny; j++) {
      fprintf(fw,"%le\t", phi[j + ny * i]);
    }
    fprintf(fw,"\n");
  }
  fclose(fw);

}
