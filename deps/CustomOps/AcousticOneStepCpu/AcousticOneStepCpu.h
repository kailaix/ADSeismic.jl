void AcousticOneStepCpuForward(const double*  w,
    const double*  wold,
    const double*  phi,
    const double*  psi,
    const double*  sigma,
    const double*  tau,
    const double*  c,
    double  dt,
    double  hx,
    double  hy,
    int64  NX,
    int64  NY,
    double*  u,
    double*  phiout,
    double*  psiout){


    for(int i = 0; i < NX+2; i++){
      for(int j = 0; j < NY+2; j++){
          int IJ = i * (NY+2) + j;
          int IpJ = (i + 1) * (NY+2) + j;
          int InJ = (i - 1) * (NY+2) + j;
          int IJp = i * (NY+2) + j + 1;
          int IJn = i * (NY+2) + j - 1;


          if (i==0 || i==NX+1 || j==0 || j==NY+1){
              u[IJ] = 0.0;
              phiout[IJ] = 0.0;
              psiout[IJ] = 0.0;
              continue;
          }

          u[IJ] = (2 - sigma[IJ]*tau[IJ]*dt*dt - 2*dt*dt/hx/hx * c[IJ] - 2*dt*dt/hy/hy * c[IJ]) * w[IJ] +
                  c[IJ] * (dt/hx)*(dt/hx)  *  (w[IpJ]+w[InJ]) +
                  c[IJ] * (dt/hy)*(dt/hy)  *  (w[IJp]+w[IJn]) +
                  (dt*dt/(2.0*hx))*(phi[IpJ]-phi[InJ]) +
                  (dt*dt/(2.0*hy))*(psi[IJp]-psi[IJn]) -
                      (1 - (sigma[IJ]+tau[IJ])*dt/2) * wold[IJ];
          u[IJ] = u[IJ] / (1 + (sigma[IJ]+tau[IJ])/2*dt);
          phiout[IJ] = (1. -dt*sigma[IJ]) * phi[IJ] + dt * c[IJ] * (tau[IJ] -sigma[IJ])/2.0/hx *  
              (w[IpJ]-w[InJ]);
          psiout[IJ] = (1. -dt*tau[IJ]) * psi[IJ] + dt * c[IJ] * (sigma[IJ] -tau[IJ])/2.0/hy * 
              (w[IJp]-w[IJn]);
      }
    }
    
}


void AcousticOneStepCpuBackward(double*  grad_w,
    double*  grad_wold,
    double*  grad_phi,
    double*  grad_psi,
    double*  grad_c,
    const double*  grad_u,
    const double*  grad_phiout,
    const double*  grad_psiout,
    const double*  w,
    const double*  wold,
    const double*  phi,
    const double*  psi,
    const double*  sigma,
    const double*  tau,
    const double*  c,
    double  dt,
    double  hx,
    double  hy,
    int64  NX,
    int64  NY,
    const double*  u,
    const double*  phiout,
    const double*  psiou){
    

    for (int i = 1; i < NX+1; i ++){
      for(int j = 1; j < NY+1; j++){
        
    
          int IJ = i * (NY+2) + j;
          int IpJ = (i + 1) * (NY+2) + j;
          int InJ = (i - 1) * (NY+2) + j;
          int IJp = i * (NY+2) + j + 1;
          int IJn = i * (NY+2) + j - 1;

          // u[IJ] = (2 - sigma[IJ]*tau[IJ]*dt*dt - 2*dt*dt/hx/hx * c[IJ] - 2*dt*dt/hy/hy * c[IJ]) * w[IJ] +
          //         c[IJ] * (dt/hx)*(dt/hx)  *  (w[IpJ]+w[InJ]) +
          //         c[IJ] * (dt/hy)*(dt/hy)  *  (w[IJp]+w[IJn]) +
          //         (dt*dt/(2.0*hx))*(phi[IpJ]-phi[InJ]) +
          //         (dt*dt/(2.0*hy))*(psi[IJp]-psi[IJn]) -
          //             (1 - (sigma[IJ]+tau[IJ])*dt/2) * wold[IJ];
          // u[IJ] = u[IJ] / (1 + (sigma[IJ]+tau[IJ])/2*dt);

          grad_c[IJ] += ((- 2*dt*dt/hx/hx - 2*dt*dt/hy/hy ) * w[IJ] + (dt/hx)*(dt/hx)  *  (w[IpJ]+w[InJ]) + 
                          (dt/hy)*(dt/hy)  *  (w[IJp]+w[IJn])) * grad_u[IJ] / (1 + (sigma[IJ]+tau[IJ])/2*dt);
          grad_w[IJ] += (2 - sigma[IJ]*tau[IJ]*dt*dt - 2*dt*dt/hx/hx * c[IJ] - 2*dt*dt/hy/hy * c[IJ]) * grad_u[IJ] / (1 + (sigma[IJ]+tau[IJ])/2*dt);
          grad_w[IpJ] += c[IJ] * (dt/hx)*(dt/hx) * grad_u[IJ] / (1 + (sigma[IJ]+tau[IJ])/2*dt);
          grad_w[InJ] += c[IJ] * (dt/hx)*(dt/hx) * grad_u[IJ] / (1 + (sigma[IJ]+tau[IJ])/2*dt);
          grad_w[IJp] +=  c[IJ] * (dt/hy)*(dt/hy) * grad_u[IJ] / (1 + (sigma[IJ]+tau[IJ])/2*dt);
          grad_w[IJn] +=  c[IJ] * (dt/hy)*(dt/hy) * grad_u[IJ] / (1 + (sigma[IJ]+tau[IJ])/2*dt);
          grad_phi[IpJ] += (dt*dt/(2.0*hx)) * grad_u[IJ] / (1 + (sigma[IJ]+tau[IJ])/2*dt);
          grad_phi[InJ] += -(dt*dt/(2.0*hx)) * grad_u[IJ] / (1 + (sigma[IJ]+tau[IJ])/2*dt);
          grad_psi[IJp] += (dt*dt/(2.0*hy)) * grad_u[IJ] / (1 + (sigma[IJ]+tau[IJ])/2*dt);
          grad_psi[IJn] += -(dt*dt/(2.0*hy)) * grad_u[IJ] / (1 + (sigma[IJ]+tau[IJ])/2*dt);
          grad_wold[IJ] += -(1 - (sigma[IJ]+tau[IJ])*dt/2) * grad_u[IJ] / (1 + (sigma[IJ]+tau[IJ])/2*dt);

      //     phiout[IJ] = (1. -dt*sigma[IJ]) * phi[IJ] + dt * c[IJ] * (tau[IJ] -sigma[IJ])/2.0/hx *  
      //     (w[IpJ]-w[InJ]);

          grad_phi[IJ] += (1. -dt*sigma[IJ]) * grad_phiout[IJ];
          grad_c[IJ] += dt *  (tau[IJ] -sigma[IJ])/2.0/hx *  (w[IpJ]-w[InJ]) * grad_phiout[IJ];
          grad_w[IpJ] += dt * c[IJ] * (tau[IJ] -sigma[IJ])/2.0/hx * grad_phiout[IJ];
          grad_w[InJ] += - dt * c[IJ] * (tau[IJ] -sigma[IJ])/2.0/hx * grad_phiout[IJ];



      // psiout[IJ] = (1. -dt*tau[IJ]) * psi[IJ] + dt * c[IJ] * (sigma[IJ] -tau[IJ])/2.0/hy * 
      //     (w[IJp]-w[IJn]);
          grad_psi[IJ] += (1. -dt*tau[IJ]) * grad_psiout[IJ];
          grad_c[IJ] += dt * (sigma[IJ] -tau[IJ])/2.0/hy * (w[IJp]-w[IJn]) * grad_psiout[IJ];
          grad_w[IJp] += dt * c[IJ] * (sigma[IJ] -tau[IJ])/2.0/hy * grad_psiout[IJ];
          grad_w[IJn] += - dt * c[IJ] * (sigma[IJ] -tau[IJ])/2.0/hy * grad_psiout[IJ];
      }
    }
}