__global__ void sharedUpdList(
        int* num_phi_d, int* nap_phi_d, int* gid_d, 
        double* phi_new_d, double* phi_old_d, 
        double pss, int ni, int nx, int ny, int nz, int rank)
    {
        extern __shared__ double shared[];
        const int tid = threadIdx.x;
        const int bid = blockIdx.x;
        int id1 = bid * blockDim.x + tid;
    
        shared[tid] = 0.0;
    
        int index = nap_phi_d[id1];
        num_phi_d[id1] = 0;
    
        for (int kk = 0; kk < index; kk += 1) {
            double pl0 = phi_new_d[kk + id1 * ni];
            int kp = gid_d[kk + id1 * ni];
    
            if (pl0 > pss) {
                int list = (int)shared[tid];
                shared[tid] += 1.0;
                gid_d[list + id1 * ni] = kp;
                phi_old_d[list + id1 * ni] = pl0;
            }
        }
        __syncthreads();
    
        int id2 = bid * blockDim.x + tid;
        num_phi_d[id2] = (int)shared[tid];
        nap_phi_d[id2] = 0;
    }  

//   sharedCalcPhi<<<blocks_xyz, threadsPerBlock, threadsPerBlock * sizeof(double)>>>(num_phi_d, nap_phi_d, gid_d, phi_new_d, phi_old_d, comp_d, sumsq_phi_d, nx, ny, nz, np, ni, del_x, del_y, del_z, del_t, Bcnst, kappaPhi, emob)
__global__ void sharedCalcPhi(
    int* num_phi_d, int* nap_phi_d, int* gid_d, 
    double* phi_new_d, double* phi_old_d, double* comp_d, double* sumsq_phi_d, 
    int nx, int ny, int nz, int np, int ni, 
    double del_x, double del_y, double del_z, double del_t, 
    double Bcnst, double kappaPhi, double emob, int rank)
{
        int e, w, n ,s, u, d, kp, kpe, kpw, kpn, kps, kpt, kpb;
        int index, inde, indw, indn, inds, indt, indb, tpind;
        double sumsq, comp0, pl0, sumx, sumy, sumz, dfdphi, gradphi;
        int tempid[nog];
        double tphie[nog], tphiw[nog], tphin[nog], tphis[nog], tphit[nog], tphib[nog];

        extern __shared__ double shared[];
        const int tid = threadIdx.x;
        const int bid = blockIdx.x;

        shared[tid] = 0.0;

        int id1 = bid * blockDim.x + tid;
        index = num_phi_d[id1];
        nap_phi_d[id1] = index;
        if (id1 >= nx * ny * nz) return;

        for (int kk = 0; kk < np; kk += 1) {
                tempid[kk] = 0;
                tphie[kk] = 0.0;
                tphiw[kk] = 0.0;
                tphin[kk] = 0.0;
                tphis[kk] = 0.0;
                tphit[kk] = 0.0;
                tphib[kk] = 0.0;
        }

        for (int kk = 0; kk < ni; kk += 1) {
         int ij = (bid * blockDim.x + tid) / nz;
         int i  = ij / ny;
         int j  = ij % ny;
         int h  = (bid * blockDim.x + tid) % nz;
                if (kk < index) {
                 kp = gid_d[kk + (h + nz * (j + i * ny)) * ni];
                 tempid[kp] = 1;
                }
        }
        __syncthreads();

        for (int kk = 0; kk < ni; kk += 1) {
         int ij = (bid * blockDim.x + tid) / nz;
         int i  = ij / ny;
         int j  = ij % ny;
         int h  = (bid * blockDim.x + tid) % nz;

         e = i + 1;
         w = i - 1;
         if (j == ny - 1) n = 0;
         else n = j + 1; 
         if (j == 0) s = ny - 1;
         else s = j - 1;
         if (h == nz - 1) u = 0;
         else u = h + 1;
         if (h == 0) d = nz - 1;
         else d = h - 1;
         if ( i > 0 && i < nx-1 ) {

         inde = num_phi_d[h + nz * (j + e * ny)];
         indw = num_phi_d[h + nz * (j + w * ny)];
         indn = num_phi_d[h + nz * (n + i * ny)];
         inds = num_phi_d[h + nz * (s + i * ny)];
         indt = num_phi_d[u + nz * (j + i * ny)];
         indb = num_phi_d[d + nz * (j + i * ny)];

                if (kk < inde) {
                kpe = gid_d[kk + (h + nz * (j + e * ny)) * ni];
                tphie[kpe] = phi_old_d[kk + (h + nz * (j + e * ny)) * ni];
                 if (tempid[kpe] == 0) {
                        tempid[kpe] = 1;
                        tpind = index + (int) shared[tid];
                        shared[tid] += 1.0;
                        gid_d[tpind + (h + nz * (j + i * ny)) * ni] = kpe;
                        phi_new_d[tpind + (h + nz * (j + i * ny)) * ni] = 0.0;

                 }
                }

                if (kk < indw) {
                kpw = gid_d[kk + (h + nz * (j + w * ny)) * ni];
                tphiw[kpw] = phi_old_d[kk + (h + nz * (j + w * ny)) * ni];
                 if (tempid[kpw] == 0) {
                        tempid[kpw] = 1;
                        tpind = index + (int) shared[tid];
                        shared[tid] += 1.0;
                        gid_d[tpind + (h + nz * (j + i * ny)) * ni] = kpw;
                        phi_new_d[tpind + (h + nz * (j + i * ny)) * ni] = 0.0;

                 }
                }
                if (kk < indn) {
                kpn = gid_d[kk + (h + nz * (n + i * ny)) * ni];
                tphin[kpn] = phi_old_d[kk + (h + nz * (n + i * ny)) * ni];
                 if (tempid[kpn] == 0) {
                        tempid[kpn] = 1;
                        tpind = index + (int) shared[tid];
                        shared[tid] += 1.0;
                        gid_d[tpind + (h + nz * (j + i * ny)) * ni] = kpn;
                        phi_new_d[tpind + (h + nz * (j + i * ny)) * ni] = 0.0;

                 }
                }

                if (kk < inds) {
                kps = gid_d[kk + (h + nz * (s + i * ny)) * ni];
                tphis[kps] = phi_old_d[kk + (h + nz * (s + i * ny)) * ni];
                 if (tempid[kps] == 0) {
                        tempid[kps] = 1;
                        tpind = index + (int) shared[tid];
                        shared[tid] += 1.0;
                        gid_d[tpind + (h + nz * (j + i * ny)) * ni] = kps;
                        phi_new_d[tpind + (h + nz * (j + i * ny)) * ni] = 0.0;

                 }
                }

                if (kk < indt) {
                kpt = gid_d[kk + (u + nz * (j + i * ny)) * ni];
                tphit[kpt] = phi_old_d[kk + (u + nz * (j + i * ny)) * ni];
                 if (tempid[kpt] == 0) {
                        tempid[kpt] = 1;
                        tpind = index + (int) shared[tid];
                        shared[tid] += 1.0;
                        gid_d[tpind + (h + nz * (j + i * ny)) * ni] = kpt;
                        phi_new_d[tpind + (h + nz * (j + i * ny)) * ni] = 0.0;

                 }
                }

                if (kk < indb) {
                kpb = gid_d[kk + (d + nz * (j + i * ny)) * ni];
                tphib[kpb] = phi_old_d[kk + (d + nz * (j + i * ny)) * ni];
                 if (tempid[kpb] == 0) {
                        tempid[kpb] = 1;
                        tpind = index + (int) shared[tid];
                        shared[tid] += 1.0;
                        gid_d[tpind + (h + nz * (j + i * ny)) * ni] = kpb;
                        phi_new_d[tpind + (h + nz * (j + i * ny)) * ni] = 0.0;

                 }
                }
          }

        }
        __syncthreads();

        int id2 = bid * blockDim.x + tid;
        nap_phi_d[id2] = index + (int) shared[tid];
        index = nap_phi_d[id2];
        sumsq = sumsq_phi_d[id2];
        comp0 = comp_d[id2];
        
        if (id2 >= nx * ny * nz) return;
        for (int kk = 0; kk < index; kk += 1) {
                pl0 = phi_old_d[kk + id2 * ni];
                kp = gid_d[kk + id2 * ni];

                sumx = tphie[kp] + tphiw[kp];
                sumy = tphin[kp] + tphis[kp];
                sumz = tphit[kp] + tphib[kp];

                dfdphi = Bcnst * (12.0 * (1.0 - comp0) * pl0 - 12.0 * (2.0 - comp0) * pl0 * pl0 + 12.0 * pl0 * sumsq);
	        gradphi = (((sumx - 2.0 * pl0)/(del_x * del_x)) + ((sumy - 2.0 * pl0)/(del_y * del_y)) + ((sumz - 2.0 * pl0)/(del_z * del_z)));

                phi_new_d[kk + id2 * ni] = pl0 - emob * del_t * (dfdphi - kappaPhi * gradphi);
        }
        __syncthreads();
    
}
//  kernelCalcMu<<<dim3(nx/8,ny/8,nz/8),dim3(8,8,8)>>>(mu_d, comp_d, sumsq_phi_d, sumcub_phi_d, nx, ny, nz, del_x, del_y, del_z, Acnst, Bcnst, kappaC);
__global__ void kernelCalcMu(double* mu_d, double* comp_d, double* sumsq_phi_d, double* sumcub_phi_d, int nx, int ny, int nz, double del_x, double del_y, double del_z, double Acnst, double Bcnst, double kappaC, int rank)
{
        int e, w, n, s, t, b;
	double sumsq, sumcub, pl0, pl1, pl2, pl3, pl4, pl5, pl6, sumx, sumy, sumz, dfdc, gradc, ki;
        int i = threadIdx.x + blockIdx.x*blockDim.x;
        int j= threadIdx.y + blockIdx.y*blockDim.y;
        int h = threadIdx.z + blockIdx.z*blockDim.z;
        int id2 = h + nz * (j+i*ny);

         e = i + 1;
         w = i - 1;
         if (j == ny - 1) n = 0;
         else n = j + 1; 
         if (j == 0) s = ny - 1;
         else s = j - 1;
         if (h == nz - 1) t = 0;
         else t = h + 1;
         if (h == 0) b = nz - 1;
         else b = h - 1;

         if (id2 >= nx*ny*nz) {
                return;
            }

         if ( i > 0 && i < nx-1 ) {

                sumsq  = sumsq_phi_d[h + nz * (j+i*ny)]; 
                sumcub = sumcub_phi_d[h + nz * (j+i*ny)]; 
                pl0    = comp_d[h + nz * (j+i*ny)];
   
                pl1 = comp_d[h + nz * (j+e*ny)];
                pl2 = comp_d[h + nz * (j+w*ny)];
                pl3 = comp_d[h + nz * (n+i*ny)];
                pl4 = comp_d[h + nz * (s+i*ny)];
                pl5 = comp_d[t + nz * (j+i*ny)];
                pl6 = comp_d[b + nz * (j+i*ny)];
                 sumx = pl1 + pl2;
                 sumy = pl3 + pl4;
                 sumz = pl5 + pl6;
   
                dfdc = 2.0 * Acnst * pl0 * (2.0 * pl0 * pl0 - 3.0 * pl0 + 1.0) + Bcnst * (2.0 * pl0 - 6.0 * sumsq + 4.0 * sumcub);
                gradc = (((sumx - 2.0 * pl0)/(del_x * del_x)) + ((sumy - 2.0 * pl0)/(del_y * del_y)) + ((sumz - 2.0 * pl0)/(del_z * del_z)));
                mu_d[h + nz * (j+i*ny)] = dfdc - kappaC * gradc;

          }
}

//  kernelCalcDiff<<<dim3(nx/8,ny/8,nz/8),dim3(8,8,8)>>>(diff_d, comp_d, sum_phi_d, sumsq_phi_d, ny, nz, Dvol, Dvap, Dsurf, Dgb);
__global__ void kernelCalcDiff(double* diff_d, double* comp_d, double* sum_phi_d, double* sumsq_phi_d, int ny, int nz, int nx, double Dvol, double Dvap, double Dsurf, double Dgb, int rank)
{
	double pl0, hphi, cterm;
        int i = threadIdx.x + blockIdx.x*blockDim.x;
        int j= threadIdx.y + blockIdx.y*blockDim.y;
        int h = threadIdx.z + blockIdx.z*blockDim.z;
        int id2 = h + nz * (j+i*ny);

                pl0    = comp_d[id2];
                if (pl0 < 1.0e-9) pl0 = 0.0;
                if (pl0 > 1.0 - 1.0e-9) pl0 = 1.0;
                hphi = pl0 * pl0 * pl0 * (10.0 - 15.0 * pl0 + 6.0 * pl0 * pl0);
                cterm = 0.5 * (sum_phi_d[id2] * sum_phi_d[id2] - sumsq_phi_d[id2]);

                diff_d[id2] = Dvol * hphi + Dvap * (1.0 - hphi) + Dsurf * pl0 * (1.0 - pl0) + Dgb * cterm;

}

//  kernelCalcComp<<<dim3(nx/8,ny/8,nz/8),dim3(8,8,8)>>>(comp_new_d, comp_d, mu_d, diff_d, nx, ny, nz, del_x, del_y, del_z, del_t);
__global__ void kernelCalcComp(double* comp_new_d, double* comp_d, double* mu_d, double* diff_d, int nx, int ny, int nz, double del_x, double del_y, double del_z, double del_t, int rank)
{
	double pl0, pl1, pl2, pl3, pl4, pl5, pl6;
        double comp0, comp1, comp2, comp3, comp4, comp5, comp6;
        double vx, vy, vz, v1, v2, v3, v4, v5, v6, gradX, gradY, gradZ;
        double cle, clw, cln, cls, clt, clb;
        double vle, vlw, vln, vls, vlt, vlb;
	double D0, D1, D2, D3, D4, D5, D6;
	double Diphalf, Dinhalf, Djphalf, Djnhalf, Dhphalf, Dhnhalf;
	double gradiphalf, gradinhalf, gradjphalf, gradjnhalf, gradhphalf, gradhnhalf;
        int e, w, n, s, t, b;
        int i = threadIdx.x + blockIdx.x*blockDim.x;
        int j= threadIdx.y + blockIdx.y*blockDim.y;
        int h = threadIdx.z + blockIdx.z*blockDim.z;
        int id2 = h + nz * (j+i*ny);

         e = i + 1;
         w = i - 1;
         if (j == ny - 1) n = 0;
         else n = j + 1; 
         if (j == 0) s = ny - 1;
         else s = j - 1;
         if (h == nz - 1) t = 0;
         else t = h + 1;
         if (h == 0) b = nz - 1;
         else b = h - 1;

         if (id2 >= nx*ny*nz) {
                return;
         }

         if ( i > 0 && i < nx-1 ) {

             comp0 = comp_d[id2];
	     comp1 = comp_d[h + nz * (j+e*ny)];
	     comp2 = comp_d[h + nz * (j+w*ny)];
	     comp3 = comp_d[h + nz * (n+i*ny)];
	     comp4 = comp_d[h + nz * (s+i*ny)];
	     comp5 = comp_d[t + nz * (j+i*ny)];
	     comp6 = comp_d[b + nz * (j+i*ny)];

	     pl0 = mu_d[h + nz * (j+i*ny)];
	     pl1 = mu_d[h + nz * (j+e*ny)];
	     pl2 = mu_d[h + nz * (j+w*ny)];
	     pl3 = mu_d[h + nz * (n+i*ny)];
	     pl4 = mu_d[h + nz * (s+i*ny)];
	     pl5 = mu_d[t + nz * (j+i*ny)];
	     pl6 = mu_d[b + nz * (j+i*ny)];

	     D0 = diff_d[h + nz * (j+i*ny)];
	     D1 = diff_d[h + nz * (j+e*ny)];
	     D2 = diff_d[h + nz * (j+w*ny)];
	     D3 = diff_d[h + nz * (n+i*ny)];
	     D4 = diff_d[h + nz * (s+i*ny)];
	     D5 = diff_d[t + nz * (j+i*ny)];
	     D6 = diff_d[b + nz * (j+i*ny)];

	     Diphalf = 0.5 * (D0 + D1);
	     Dinhalf = 0.5 * (D0 + D2);
	     Djphalf = 0.5 * (D0 + D3);
	     Djnhalf = 0.5 * (D0 + D4);
	     Dhphalf = 0.5 * (D0 + D5);
	     Dhnhalf = 0.5 * (D0 + D6);

	     gradiphalf = (pl1 - pl0)/del_x;
	     gradinhalf = (pl0 - pl2)/del_x;
	     gradjphalf = (pl3 - pl0)/del_y;
	     gradjnhalf = (pl0 - pl4)/del_y;
	     gradhphalf = (pl5 - pl0)/del_z;
	     gradhnhalf = (pl0 - pl6)/del_z;

	     comp_new_d[id2] = comp0 + del_t * (((Diphalf * gradiphalf - Dinhalf * gradinhalf)/del_x) + ((Djphalf * gradjphalf - Djnhalf * gradjnhalf)/del_y) + ((Dhphalf * gradhphalf - Dhnhalf * gradhnhalf)/del_z));
          }
}

// kernelIntPhi<<<dim3(blocks,1,1),dim3(threadsPerBlock_X,threadsPerBlock_Y,1)>>>(phi_new_d);
__global__ void kernelIntPhi(double* phi_new_d, int nx, int ny, int nz, int ni, int rank)

{
    int idx= (threadIdx.x+threadIdx.y*blockDim.x) +
             (blockIdx.x*blockDim.x*blockDim.y);

        if (idx >= nx*ny*nz*ni) {
            return;
        }

       phi_new_d[idx] = 0.0;
}

__global__ void kernelUpdComp(double* phi_new_d, double* phi_old_d, int nx, int ny, int nz, int rank)

{
    int idx= (threadIdx.x+threadIdx.y*blockDim.x) +
             (blockIdx.x*blockDim.x*blockDim.y);
        
             if (idx >= nx*ny*nz) {
                return;
        }
                phi_old_d[idx] = phi_new_d[idx];
                phi_new_d[idx] = 0.0;
}

//  sharedSum<<<blocks_xyz, threadsPerBlock, threadsPerBlock * sizeof(double)>>>(num_phi_d, phi_old_d, sum_phi_d, ni, pss);
__global__ void sharedSum(int* numact_d, double * num, double * result, int ni, double pss)
{
        extern __shared__ double shared[];
        const int tid = threadIdx.x;
        const int bid = blockIdx.x;

        shared[tid] = 0;
        int id1 = bid * blockDim.x + tid;
        int index = numact_d[id1];  

        for (int i = 0; i < index; i += 1) {
        int j = (bid * blockDim.x + tid) * ni + i;
                if (num[j] > pss) shared[tid] += num[j];
        }
        __syncthreads();

        int k = bid * blockDim.x + tid;
        result[k] = shared[tid];
        if (result[k] > 1.0) result[k] = 1.0;

//      __syncthreads();
}

//  sharedSumsq<<<blocks_xyz, threadsPerBlock, threadsPerBlock * sizeof(double)>>>(num_phi_d, phi_old_d, sumsq_phi_d, ni, pss);
__global__ void sharedSumsq(int* numact_d, double * num, double * result, int ni, double pss)
{
        extern __shared__ double shared[];
        const int tid = threadIdx.x;
        const int bid = blockIdx.x;


        shared[tid] = 0;
        int id1 = bid * blockDim.x + tid;
        int index = numact_d[id1];  

        for (int i = 0; i < index; i += 1) {
        int j = (bid * blockDim.x + tid) * ni + i;
                if (num[j] > pss) shared[tid] += num[j] * num[j];
        }
        __syncthreads();

        int k = bid * blockDim.x + tid;
        result[k] = shared[tid];

//      __syncthreads();
}

//  sharedSumcub<<<blocks_xyz, threadsPerBlock, threadsPerBlock * sizeof(double)>>>(num_phi_d, phi_old_d, sumcub_phi_d, ni, pss);
__global__ void sharedSumcub(int* numact_d, double * num, double * result, int ni, double pss)
{
        extern __shared__ double shared[];
        const int tid = threadIdx.x;
        const int bid = blockIdx.x;


        shared[tid] = 0;
        int id1 = bid * blockDim.x + tid;
        int index = numact_d[id1];  

        for (int i = 0; i < index; i += 1) {
        int j = (bid * blockDim.x + tid) * ni + i;
                if (num[j] > pss) shared[tid] += num[j] * num[j] * num[j];
        }
        __syncthreads();

        int k = bid * blockDim.x + tid;
        result[k] = shared[tid];
        if (result[k] > 1.0) result[k] = 1.0;

//      __syncthreads();
}

