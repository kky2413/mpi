void Output_Conf(double *phi, double *comp, int nx, int ny,int ni, int P, char *argv[], int rank) { // 함수 정의 (Define function)

    FILE *fp1, *fp2, *fp3; // 출력 파일 포인터들 선언 (Declare file pointers for output)
    char fname1[100], fname2[100], fname3[100];
    double *phi_old_d, *phi_new_d;
    double *comp_d, *comp_new_d;
  
    int i,j,k,l;
    double phimax;
    int nmax;

    // count값을 파일 이름에 포함하는 파일 생성 (Generate file names that include the count value)
    sprintf(fname1,"comp_%08d_%d.dat", P, rank);
    sprintf(fname2,"gind_%08d_%d.dat", P, rank);
    sprintf(fname3,"prof_%08d_%d.dat", P, rank);

    // 출력 파일을 쓰기모드로 열기 (Open output files in write mode)
    fp1 = fopen(fname1, "w");
    fp2 = fopen(fname2, "w");
    fp3 = fopen(fname3, "w");

    for (i = 0; i < nx; i++) {
        for (j = 0; j < ny; j++) {
            phimax = 0.0;
            nmax = 0; // phi 최댓값을 찾기위해 초기화 시킴 (Initialize to find the maximum phi value)
            for (k = 0; k < ni; k++) {
                if (phi[k + (j + i * ny) * ni] > phimax) {
                    phimax = phi[k + (j + i * ny) * ni]; // 최댓값보다 클 경우 계속 업데이트 됨 (Update if greater than the current maximum)
                    nmax = k;
                }
                if (phimax < 1.0e-8) nmax = -1; // 최댓값이 너무 작으면 nmax를 -1로 하므로써 멈춤 (Set nmax to -1 if the maximum is too small)
            }
            fprintf(fp1, "%le\t", comp[j + i * ny]); // comp 값을 fp1 파일에 씀 (Write comp value to fp1 file)
            fprintf(fp2, "%d\t", nmax); // nmax 값을 fp2 파일에 씀 (Write nmax value to fp2 file)
            fprintf(fp3, "%lf\t", phimax); // phimax 값을 fp3 파일에 씀 (Write phimax value to fp3 file)
        }
        fprintf(fp1, "\n"); 
        fprintf(fp2, "\n"); 
        fprintf(fp3, "\n"); 
    }

    // 파일 닫기 (Close files)
    fclose(fp1);
    fclose(fp2);
    fclose(fp3);
}
