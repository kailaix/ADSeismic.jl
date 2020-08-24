#define EIGEN_USE_GPU
#include "cuda.h"
#include <stdio.h>

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



__global__ void Init(const long long size, const double* in, double* out) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if(i<size) out[i] = in[i];
}

__global__ void AddSource(
    double * vx_,
    double * vy_,
    double * sigmaxx_,
    double * sigmayy_,
    double * sigmaxy_,
    const long long  *srci, const long long  *srcj, const double *srcv, const long long  *src_type, const int nsrc,
    long long NX, long long NY) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    // printf("%d\n", i);
    if (i<nsrc){
      switch (src_type[i])
        {
            case 0: // vx
              vx_(srci[i]-1, srcj[i]-1) = srcv[i];
              break;

            case 1: // vy
              vy_(srci[i]-1, srcj[i]-1) = srcv[i];
              break;

            case 2:
              sigmaxx_(srci[i]-1, srcj[i]-1) = srcv[i];
              // printf("add source to xx, %d\n", i);
              break;

            case 3:
              sigmayy_(srci[i]-1, srcj[i]-1) = srcv[i];
              // printf("add source to yy, %d\n", i);
              break;

            case 4:
              sigmaxy_(srci[i]-1, srcj[i]-1) = srcv[i];
              break;
        
        default:
          break;
      }
    }
}

__global__ void SetSourceGrad(
  double * g_vx_,
  double * g_vy_,
  double * g_sigmaxx_,
  double * g_sigmayy_,
  double * g_sigmaxy_,
  double * grad_srcv,
  const long long  *srci, const long long  *srcj, const double *srcv, const long long  *src_type, const int nsrc,
  long long NX, long long NY) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  // printf("%d\n", i);
  if (i<nsrc){
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


void forwardGPU(
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
    const long long  * srci, const long long  *srcj, const double *srcv, const long long  *src_type, long long  nsrc,
    const long long *    nx, const long long *    ny
){
  // printf("FORWARD GPU\n");
  long long NX, NY;
  cudaMemcpy(&NX, nx, sizeof(long long), cudaMemcpyDeviceToHost);
  cudaMemcpy(&NY, ny, sizeof(long long), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  // printf("Running on the GPU\n");
  // printf("%d\n",__LINE__);
  Init<<<((NX+2)*(NY+2)+255)/256, 256>>>((NX+2)*(NY+2), vx, vx_);
  // printf("%d\n",__LINE__);
  Init<<<((NX+2)*(NY+2)+255)/256, 256>>>((NX+2)*(NY+2), vy, vy_);
  // printf("%d\n",__LINE__);
  Init<<<((NX+2)*(NY+2)+255)/256, 256>>>((NX+2)*(NY+2), sigmaxx, sigmaxx_);
  // printf("%d\n",__LINE__);
  Init<<<((NX+2)*(NY+2)+255)/256, 256>>>((NX+2)*(NY+2), sigmayy, sigmayy_);
  // printf("%d\n",__LINE__);
  Init<<<((NX+2)*(NY+2)+255)/256, 256>>>((NX+2)*(NY+2), sigmaxy, sigmaxy_);  
  // printf("%d\n",__LINE__);

  // cudaError_t cudaerr = cudaDeviceSynchronize();
  //   if (cudaerr != cudaSuccess)
  //   {
  //       printf("kernel launch failed with error \"%s\".\n", cudaGetErrorString(cudaerr));
  //   }

  AddSource<<<(nsrc+255)/256, 256>>>(vx_, vy_, sigmaxx_, sigmayy_, sigmaxy_, srci, srcj, srcv, src_type, nsrc, NX, NY);
  // printf("%d\n",__LINE__);

  // cudaerr = cudaDeviceSynchronize();
  //   if (cudaerr != cudaSuccess)
  //   {
  //       printf("kernel launch failed with error \"%s\".\n", cudaGetErrorString(cudaerr));
  //   }
}



void backwardGPU(
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
    const long long  * srci, const long long  *srcj, const double *srcv, const long long  *src_type, long long  nsrc,
    const long long *    nx, const long long *    ny
){
  // printf("BACKWARD GPU\n");
  long long NX, NY;
  cudaMemcpy(&NX, nx, sizeof(long long), cudaMemcpyDeviceToHost);
  cudaMemcpy(&NY, ny, sizeof(long long), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  Init<<<((NX+2)*(NY+2)+255)/256, 256>>>((NX+2)*(NY+2), g_sigmaxx_, g_sigmaxx);
  Init<<<((NX+2)*(NY+2)+255)/256, 256>>>((NX+2)*(NY+2), g_sigmaxy_, g_sigmaxy);
  Init<<<((NX+2)*(NY+2)+255)/256, 256>>>((NX+2)*(NY+2), g_sigmayy_, g_sigmayy);
  Init<<<((NX+2)*(NY+2)+255)/256, 256>>>((NX+2)*(NY+2), g_vx_, g_vx);
  Init<<<((NX+2)*(NY+2)+255)/256, 256>>>((NX+2)*(NY+2), g_vy_, g_vy);


  SetSourceGrad<<<nsrc/256, 256>>>(g_vx, g_vy, g_sigmaxx, g_sigmayy, g_sigmaxy, grad_srcv, 
    srci, srcj, srcv, src_type, nsrc, NX, NY);

}

