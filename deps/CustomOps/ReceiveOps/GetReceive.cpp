#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"
#include<cmath>
#include<string> 
using std::string;
using namespace tensorflow;

void forward(double *out, const double*vx, const double*vy, 
      const double*sigmaxx, const double*sigmayy, const double*sigmaxy, int64 nt,
      const int64 *rcvi, const int64 *rcvj, const int64 *rcvtype, int64 nrcv, int64 NX, int64 NY){
    for(int i=0;i<nrcv;i++){
      int idx = (rcvi[i]-1)*(NY+2) + rcvj[i]-1;
         switch (rcvtype[i])
         {
         case 0:
           for(int k=0;k<nt;k++)
              out[nt*i+k] = vx[k*(NX+2)*(NY+2)+idx];
           break;

          case 1:
            for(int k=0;k<nt;k++)
              out[nt*i+k] = vy[k*(NX+2)*(NY+2)+idx];
           break;

           case 2:
            for(int k=0;k<nt;k++)
              out[nt*i+k] = sigmaxx[k*(NX+2)*(NY+2)+idx];
           break;

           case 3:
            for(int k=0;k<nt;k++)
              out[nt*i+k] = sigmayy[k*(NX+2)*(NY+2)+idx];
           break;

           case 4:
            for(int k=0;k<nt;k++)
              out[nt*i+k] = sigmaxy[k*(NX+2)*(NY+2)+idx];
           break;
         
         default:
           break;
         }
    }
}

void backward(
    double*d_vx, double*d_vy, 
    double*d_sigmaxx, double*d_sigmayy, double*d_sigmaxy, const double *d_out, 
    int64 nt, const int64 *rcvi, const int64 *rcvj, const int64 *rcvtype, int64 nrcv, int64 NX, int64 NY){
  for(int i=0;i<nt*(NX+2)*(NY+2);i++){
    d_vx[i] = 0.0;
    d_vy[i] = 0.0;
    d_sigmaxx[i] = 0.0;
    d_sigmayy[i] = 0.0;
    d_sigmaxy[i] = 0.0;
  }

  for(int i=0;i<nrcv;i++){
      int idx = (rcvi[i]-1)*(NY+2) + rcvj[i]-1;
         switch (rcvtype[i])
         {
         case 0:
           for(int k=0;k<nt;k++)
              d_vx[k*(NX+2)*(NY+2)+idx] += d_out[nt*i+k];
           break;

          case 1:
            for(int k=0;k<nt;k++){
              // printf("Top gradients: %f\n", d_out[nt*i+k]);
              d_vy[k*(NX+2)*(NY+2)+idx] += d_out[nt*i+k];
            }
              
           break;

           case 2:
            for(int k=0;k<nt;k++)
              d_sigmaxx[k*(NX+2)*(NY+2)+idx] += d_out[nt*i+k];
           break;

           case 3:
            for(int k=0;k<nt;k++)
              d_sigmayy[k*(NX+2)*(NY+2)+idx] += d_out[nt*i+k];
           break;

           case 4:
            for(int k=0;k<nt;k++)
              d_sigmaxy[k*(NX+2)*(NY+2)+idx] += d_out[nt*i+k];
           break;
         
         default:
           break;
         }
    }
    
}