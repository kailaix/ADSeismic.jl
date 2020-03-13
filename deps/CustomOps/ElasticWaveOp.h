#define lambda(i,j) lambda[(i)*(NY+2)+(j)]
#define mu(i,j) mu[(i)*(NY+2)+(j)]
#define rho(i,j) rho[(i)*(NY+2)+(j)]
#define vx(i,j) vx[(i)*(NY+2)+(j)]
#define vy(i,j) vy[(i)*(NY+2)+(j)]
#define vx_(i,j) vx_[(i)*(NY+2)+(j)]
#define vy_(i,j) vy_[(i)*(NY+2)+(j)]
#define sigmaxx(i,j) sigmaxx[(i)*(NY+2)+(j)]
#define sigmaxx_(i,j) sigmaxx_[(i)*(NY+2)+(j)]
#define sigmayy(i,j) sigmayy[(i)*(NY+2)+(j)]
#define sigmayy_(i,j) sigmayy_[(i)*(NY+2)+(j)]
#define sigmaxy(i,j) sigmaxy[(i)*(NY+2)+(j)]
#define sigmaxy_(i,j) sigmaxy_[(i)*(NY+2)+(j)]
#define memory_dvx_dx_(i,j) memory_dvx_dx_[(i)*(NY+2)+(j)]
#define memory_dvx_dy_(i,j) memory_dvx_dy_[(i)*(NY+2)+(j)]
#define memory_dvy_dx_(i,j) memory_dvy_dx_[(i)*(NY+2)+(j)]
#define memory_dvy_dy_(i,j) memory_dvy_dy_[(i)*(NY+2)+(j)]
#define memory_dsigmaxx_dx_(i,j) memory_dsigmaxx_dx_[(i)*(NY+2)+(j)]
#define memory_dsigmayy_dy_(i,j) memory_dsigmayy_dy_[(i)*(NY+2)+(j)]
#define memory_dsigmaxy_dx_(i,j) memory_dsigmaxy_dx_[(i)*(NY+2)+(j)]
#define memory_dsigmaxy_dy_(i,j) memory_dsigmaxy_dy_[(i)*(NY+2)+(j)]
#define memory_dvx_dx(i,j) memory_dvx_dx[(i)*(NY+2)+(j)]
#define memory_dvx_dy(i,j) memory_dvx_dy[(i)*(NY+2)+(j)]
#define memory_dvy_dx(i,j) memory_dvy_dx[(i)*(NY+2)+(j)]
#define memory_dvy_dy(i,j) memory_dvy_dy[(i)*(NY+2)+(j)]
#define memory_dsigmaxx_dx(i,j) memory_dsigmaxx_dx[(i)*(NY+2)+(j)]
#define memory_dsigmayy_dy(i,j) memory_dsigmayy_dy[(i)*(NY+2)+(j)]
#define memory_dsigmaxy_dx(i,j) memory_dsigmaxy_dx[(i)*(NY+2)+(j)]
#define memory_dsigmaxy_dy(i,j) memory_dsigmaxy_dy[(i)*(NY+2)+(j)]

void forward(
    double * vx_,
    double * vy_,
    double * sigmaxx_,
    double * sigmayy_,
    double * sigmaxy_,
    const double * lambda,
    const double * mu,
    const double * rho,
    double * memory_dvx_dx_,
    double * memory_dvx_dy_,
    double * memory_dvy_dx_,
    double * memory_dvy_dy_,
    double * memory_dsigmaxx_dx_,
    double * memory_dsigmayy_dy_,
    double * memory_dsigmaxy_dx_,
    double * memory_dsigmaxy_dy_,
    const double * vx,
    const double * vy,
    const double * sigmaxx,
    const double * sigmayy,
    const double * sigmaxy,
    const double * memory_dvx_dx,
    const double * memory_dvx_dy,
    const double * memory_dvy_dx,
    const double * memory_dvy_dy,
    const double * memory_dsigmaxx_dx,
    const double * memory_dsigmayy_dy,
    const double * memory_dsigmaxy_dx,
    const double * memory_dsigmaxy_dy,
    const double * K_x,   // PML parameter
    const double * K_y,
    const double * K_x_half,
    const double * K_y_half,
    const double * b_x,
    const double * b_y,
    const double * b_x_half,
    const double * b_y_half,
    const double * a_x,
    const double * a_y,
    const double * a_x_half,
    const double * a_y_half,
    const double * alpha_x,
    const double * alpha_y,
    const double * alpha_x_half,
    const double * alpha_y_half,
    const int64 * srci, const int64 *srcj, const double *srcv, const int64 *src_type, int64 nsrc,
    double DELTAX, double DELTAY, double DELTAT,
    int64 NX, int64 NY
){

  for(int i=0;i<NX+2;i++)
    for(int j=0;j<NY+2;j++){
      sigmaxx_(i,j) = 0.0;
      sigmaxy_(i,j) = 0.0;
      sigmayy_(i,j) = 0.0;
      vx_(i,j) = 0.0;
      vy_(i,j) = 0.0;
    }

  for(int j=2;j<NY+1;j++)
    for(int i=1;i<NX;i++){
      double lambda_half_x = 0.5 * (lambda(i+1,j) + lambda(i,j));
      double mu_half_x = 0.5 * (mu(i+1,j) + mu(i,j));
      double lambda_plus_two_mu_half_x = lambda_half_x + 2. * mu_half_x;

      double value_dvx_dx = (27*vx(i+1,j)-27.*vx(i,j)-vx(i+2,j)+vx(i-1,j)) / (24.*DELTAX);
      double value_dvy_dy = (27*vy(i,j)-27.*vy(i,j-1)-vy(i,j+1)+vy(i,j-2)) / (24.*DELTAY);

      memory_dvx_dx_(i,j) = b_x_half[i-1] * memory_dvx_dx(i,j) + a_x_half[i-1] * value_dvx_dx;
      memory_dvy_dy_(i,j) = b_y[j-1] * memory_dvy_dy(i,j) + a_y[j-1] * value_dvy_dy;

      value_dvx_dx = value_dvx_dx / K_x_half[i-1] + memory_dvx_dx(i,j);
      value_dvy_dy = value_dvy_dy / K_y[j-1] + memory_dvy_dy(i,j);
      

      sigmaxx_(i,j) = sigmaxx(i,j) + 
        (lambda_plus_two_mu_half_x * value_dvx_dx + lambda_half_x * value_dvy_dy) * DELTAT;

      sigmayy_(i,j) = sigmayy(i,j) + 
        (lambda_half_x * value_dvx_dx + lambda_plus_two_mu_half_x * value_dvy_dy) * DELTAT;
    }
  

  for(int j=1;j<NY;j++){
    for(int i=2;i<NX+1;i++){
      double mu_half_y = 0.5 * (mu(i,j+1) + mu(i,j));

      double value_dvy_dx = (27.*vy(i,j)-27.*vy(i-1,j)-vy(i+1,j)+vy(i-2,j)) / (24.*DELTAX);
      double value_dvx_dy = (27.*vx(i,j+1)-27.*vx(i,j)-vx(i,j+2)+vx(i,j-1)) / (24.*DELTAY);

      memory_dvy_dx_(i,j) = b_x[i-1] * memory_dvy_dx(i,j) + a_x[i-1] * value_dvy_dx;
      memory_dvx_dy_(i,j) = b_y_half[j-1] * memory_dvx_dy(i,j) + a_y_half[j-1] * value_dvx_dy;

      value_dvy_dx = value_dvy_dx / K_x[i-1] + memory_dvy_dx(i,j);
      value_dvx_dy = value_dvx_dy / K_y[j-1] + memory_dvx_dy(i,j);

      sigmaxy_(i,j) = sigmaxy(i,j) + mu_half_y * (value_dvy_dx + value_dvx_dy) * DELTAT;
    }
  }
   
  for(int j=2;j<NY+1;j++)
    for(int i=2;i<NX+1;i++){
      double value_dsigmaxx_dx = (27.*sigmaxx_(i,j)-27.*sigmaxx_(i-1,j)-sigmaxx_(i+1,j)+sigmaxx_(i-2,j)) / (24.*DELTAX);
      double value_dsigmaxy_dy = (27.*sigmaxy_(i,j)-27.*sigmaxy_(i,j-1)-sigmaxy_(i,j+1)+sigmaxy_(i,j-2)) / (24.*DELTAY);

      memory_dsigmaxx_dx_(i,j) = b_x[i-1] * memory_dsigmaxx_dx(i,j) + a_x[i-1] * value_dsigmaxx_dx;
      memory_dsigmaxy_dy_(i,j) = b_y[j-1] * memory_dsigmaxy_dy(i,j) + a_y[j-1] * value_dsigmaxy_dy;

      value_dsigmaxx_dx = value_dsigmaxx_dx / K_x[i-1] + memory_dsigmaxx_dx(i,j);
      value_dsigmaxy_dy = value_dsigmaxy_dy / K_y[j-1] + memory_dsigmaxy_dy(i,j);

      vx_(i,j) = vx(i,j) + (value_dsigmaxx_dx + value_dsigmaxy_dy) * DELTAT / rho(i,j);
  }

  for(int j=1;j<NY;j++){
    for(int i=1;i<NX;i++){
      double rho_half_x_half_y = 0.25 * (rho(i,j) + rho(i+1,j) + rho(i+1,j+1) + rho(i,j+1));

      double value_dsigmaxy_dx = (27.*sigmaxy_(i+1,j)-27.*sigmaxy_(i,j)-sigmaxy_(i+2,j)+sigmaxy_(i-1,j)) / (24.*DELTAX);
      double value_dsigmayy_dy = (27.*sigmayy_(i,j+1)-27.*sigmayy_(i,j)-sigmayy_(i,j+2)+sigmayy_(i,j-1)) / (24.*DELTAY);
      // printf("DEBUG: %f %f %f\n", value_dsigmaxy_dx, value_dsigmayy_dy, rho_half_x_half_y);

      memory_dsigmaxy_dx_(i,j) = b_x_half[i-1] * memory_dsigmaxy_dx(i,j) + a_x_half[i-1] * value_dsigmaxy_dx;
      memory_dsigmayy_dy_(i,j) = b_y_half[j-1] * memory_dsigmayy_dy(i,j) + a_y_half[j-1] * value_dsigmayy_dy;

      value_dsigmaxy_dx = value_dsigmaxy_dx / K_x_half[i-1] + memory_dsigmaxy_dx(i,j);
      value_dsigmayy_dy = value_dsigmayy_dy / K_y_half[j-1] + memory_dsigmayy_dy(i,j);

      // printf("DEBUG: %f %f %f\n", value_dsigmaxy_dx, value_dsigmayy_dy, rho_half_x_half_y);
      vy_(i,j) = vy(i,j) + (value_dsigmaxy_dx + value_dsigmayy_dy) * DELTAT / rho_half_x_half_y;
    }
  }

  for(int i=0;i<nsrc;i++){
      switch (src_type[i])
        {
            case 0: // vx
              vx_(srci[i], srcj[i]) += srcv[i];
              break;

            case 1: // vy
              vy_(srci[i], srcj[i]) += srcv[i];
              break;

            case 2:
              sigmaxx_(srci[i], srcj[i]) += srcv[i];
              // printf("add source to xx, %d\n", i);
              break;

            case 3:
              sigmayy_(srci[i], srcj[i]) += srcv[i];
              // printf("add source to yy, %d\n", i);
              break;

            case 4:
              sigmaxy_(srci[i], srcj[i]) += srcv[i];
              break;
        
        default:
          break;
        }
  }

}
