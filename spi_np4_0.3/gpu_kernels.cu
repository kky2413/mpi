
__global__ void kernelCalcMu (double* mu_d,  double* phi_old_d, int nx, int ny, double del_x, double del_y, double Acnst, double kappa, int rank)
{
    int e, w, n, s;
    double pl0, pl1, pl2, pl3, pl4, dfdphi, grad2phi;
    int i= threadIdx.x + blockIdx.x*blockDim.x;
    int j= threadIdx.y + blockIdx.y*blockDim.y;
    int id = j + i * ny;

    if (id >= nx*ny){
      return;
    }
        
        e = i + 1;
        w = i - 1;
        if (j == ny - 1) n = 0;
        else n = j + 1; 
        if (j == 0) s = ny - 1;
        else s = j - 1;

        if (i > 0 && i< nx-1) {

             pl0 = phi_old_d[id];

             pl1 = phi_old_d[j+e*ny];
             pl2 = phi_old_d[j+w*ny];
             pl3 = phi_old_d[n+i*ny];
             pl4 = phi_old_d[s+i*ny];

	     dfdphi = 2.0*Acnst*pl0*(1.0-pl0)*(1.0-2.0*pl0);
	     grad2phi = (((pl1+pl2-2.0*pl0)/(del_x * del_x)) + ((pl3+pl4-2.0*pl0)/(del_y * del_y)));

	     mu_d[j+i*ny] =dfdphi - (2.0 * kappa * grad2phi);

        }
}

__global__ void kernelCalCom (double* mu_d, double* phi_new_d,  double* phi_old_d, int nx, int ny, double del_x, double del_y, double del_t, double atmob, int rank)
{
    int e, w, n, s;
    double pl0, mu0, mu1, mu2, mu3, mu4, fx1,fx2,fy1,fy2;
    int i= threadIdx.x + blockIdx.x*blockDim.x;
    int j= threadIdx.y + blockIdx.y*blockDim.y;
    int id = j + i * ny;
    
    if ( id >= nx*ny) {
        return;
    }

        e = i + 1;
        w = i - 1;
        if (j == ny - 1) n = 0;
        else n = j + 1;
        if (j == 0) s = ny - 1;
        else s = j - 1;

        if (i > 0 && i<nx-1) {

             pl0 = phi_old_d[id];


             mu0 = mu_d[id];
             mu1 = mu_d[j+e*ny];
             mu2 = mu_d[j+w*ny];
             mu3 = mu_d[n+i*ny];
             mu4 = mu_d[s+i*ny];

             fx1 = -atmob*((mu1 - mu0)/del_x);
             fx2 = -atmob*((mu0 - mu2)/del_x);
             fy1 = -atmob*((mu3 - mu0)/del_y);
             fy2 = -atmob*((mu0 - mu4)/del_y);


             phi_new_d[j+i*ny] = pl0 - del_t*(((fx1 - fx2)/del_x)+((fy1 - fy2)/del_y));

          }
}

__global__ void kernelUpdPhi(double* phi_new_d, double* phi_old_d, int nx, int ny, int rank)
{

    int i= threadIdx.x + blockIdx.x*blockDim.x;
    int j= threadIdx.y + blockIdx.y*blockDim.y;
    int id = j + i * ny;
 
     if (id < nx*ny){
         phi_old_d[id] = phi_new_d[id];
    }     
}

