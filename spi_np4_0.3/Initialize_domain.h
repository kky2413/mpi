void Initialize_domain(double *phi, int radius, double del_x, int nx, int ny, int Nx, int Ny, int rank, long SEED, double alloycomp, double noise) {
    int i, j, index, ig;
    int x0, y0;
    double *psi = (double *)malloc(Nx * Ny * sizeof(double));  // 동적 메모리 할당
    if (psi == NULL) {
        fprintf(stderr, "Memory allocation failed!\n");
        exit(EXIT_FAILURE);
    }

    srand(SEED+rank);

    for (i = 0; i < Nx; i++) {
        for (j = 0; j < Ny; j++) {
            psi[j + i * Ny] = alloycomp + (2.0 * (double)rand() / RAND_MAX - 1.0) * noise;
        }
    }

    // 중심 좌표 (x0, y0)
    //x0 = Nx / 2 + 1;
    //y0 = Ny / 2;

    // nx = Nx/nprocs + 2 이므로 x좌표의 시작점이 1부터 nx-1까지이다
    for (i = 1; i < nx - 1; i++) {
        for (j = 0; j < ny; j++) {
            index = j + i * ny;
            ig = i - 1 + rank * (nx - 2);  // 글로벌 인덱스 ig를 계산
            phi[index] = psi[j + ig * Ny];
        }
    }

    free(psi);  // 동적 메모리 해제
}

