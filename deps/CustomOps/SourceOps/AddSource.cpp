#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"
#include<cmath>
#include<string> 
using std::string;
using namespace tensorflow;

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

#define g_vx(i,j) g_vx[(i)*(NY+2)+(j)]
#define g_vy(i,j) g_vy[(i)*(NY+2)+(j)]
#define g_vx_(i,j) g_vx_[(i)*(NY+2)+(j)]
#define g_vy_(i,j) g_vy_[(i)*(NY+2)+(j)]
#define g_sigmaxx(i,j) g_sigmaxx[(i)*(NY+2)+(j)]
#define g_sigmaxx_(i,j) g_sigmaxx_[(i)*(NY+2)+(j)]
#define g_sigmayy(i,j) g_sigmayy[(i)*(NY+2)+(j)]
#define g_sigmayy_(i,j) g_sigmayy_[(i)*(NY+2)+(j)]
#define g_sigmaxy(i,j) g_sigmaxy[(i)*(NY+2)+(j)]
#define g_sigmaxy_(i,j) g_sigmaxy_[(i)*(NY+2)+(j)]


void forwardCPU(
    double * vx_,
    double * vy_,
    double * sigmaxx_,
    double * sigmayy_,
    double * sigmaxy_,
    const double * vx,
    const double * vy,
    const double * sigmaxx,
    const double * sigmayy,
    const double * sigmaxy,
    const int64 * srci, const int64 *srcj, const double *srcv, const int64 *src_type, int64 nsrc,
    int64 NX, int64 NY
){
  // printf("Running on the CPU\n");
  for(int i=0;i<NX+2;i++)
    for(int j=0;j<NY+2;j++){
      sigmaxx_(i,j) = sigmaxx(i,j);
      sigmaxy_(i,j) = sigmaxy(i,j);
      sigmayy_(i,j) = sigmayy(i,j);
      vx_(i,j) = vx(i,j);
      vy_(i,j) = vy(i,j);
  }

  for(int i=0;i<nsrc;i++){
      switch (src_type[i])
        {
            case 0: // vx
              vx_(srci[i]-1, srcj[i]-1) += srcv[i];
              break;

            case 1: // vy
              vy_(srci[i]-1, srcj[i]-1) += srcv[i];
              break;

            case 2:
              sigmaxx_(srci[i]-1, srcj[i]-1) += srcv[i];
              // printf("add source to xx, %d\n", i);
              break;

            case 3:
              sigmayy_(srci[i]-1, srcj[i]-1) += srcv[i];
              // printf("add source to yy, %d\n", i);
              break;

            case 4:
              sigmaxy_(srci[i]-1, srcj[i]-1) += srcv[i];
              break;
        
        default:
          break;
        }
  }

}



void backwardCPU(
    double * g_vx,
    double * g_vy,
    double * g_sigmaxx,
    double * g_sigmayy,
    double * g_sigmaxy,
    double * grad_srcv,
    const double * g_vx_,
    const double * g_vy_,
    const double * g_sigmaxx_,
    const double * g_sigmayy_,
    const double * g_sigmaxy_,
    const int64 * srci, const int64 *srcj, const double *srcv, const int64 *src_type, int64 nsrc,
    int64 NX, int64 NY
){
// printf("DEBUG: \n");
  for(int i=0;i<NX+2;i++)
    for(int j=0;j<NY+2;j++){
      g_sigmaxx(i,j) = g_sigmaxx_(i,j);
      g_sigmaxy(i,j) = g_sigmaxy_(i,j);
      g_sigmayy(i,j) = g_sigmayy_(i,j);
      g_vx(i,j) = g_vx_(i,j);
      g_vy(i,j) = g_vy_(i,j);
  }

  // string str;
  // char ss[1024];
  // for (int i = 0; i<NX + 2; i++){
  //   for(int j = 0; j<NY + 2; j++){
  //     sprintf(ss, "%f ", g_vx(i, j));
  //     str += string(ss);
  //   }
  //   str += "\n";
  // }
  // str += "-------------------------------------------------";
  // printf("%s\n", str.c_str());

  
  for(int i=0;i<nsrc;i++){
      switch (src_type[i])
        {
            case 0: // vx
              grad_srcv[i] = g_vx_(srci[i]-1, srcj[i]-1);
              break;

            case 1: // vy
              grad_srcv[i] = g_vy_(srci[i]-1, srcj[i]-1);
              break;

            case 2:
              grad_srcv[i] = g_sigmaxx_(srci[i]-1, srcj[i]-1);
              break;

            case 3:
              grad_srcv[i] = g_sigmayy_(srci[i]-1, srcj[i]-1);
              break;

            case 4:
              grad_srcv[i] = g_sigmaxy_(srci[i]-1, srcj[i]-1);
              break;
        
        default:
          break;
        }
  }

  

}