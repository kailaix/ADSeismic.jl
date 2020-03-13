#include "cuda.h"
typedef long long int64;
__global__ void Init(double *out, const double *in, int size){
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if(i<size) out[i] = in[i];
}

__global__ void Zero(double *out, int size){
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if(i<size) out[i] = 0.0;
}

__global__ void AddSource(double *out, const int64 *srci, const int64* srcj, const double *srcv, int nsrc, int64 NY){
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if(i<nsrc){
    // printf("source: %d, %d, (%d, %d), %f\n", srci[i], srcj[i], NX, NY, srcv[i]);
    int idx = (srci[i]-1)*(NY+2)+srcj[i]-1;
    out[idx] += srcv[i];
  }
}

__global__ void AddSourceGrad(double *grad_src, const int64 *srci, const int64* srcj, const double *grad_out, int nsrc, int64 NY){
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if(i<nsrc){
    int idx = (srci[i]-1)*(NY+2)+srcj[i]-1;
    grad_src[i] += grad_out[idx];
  }
}

void forwardGPU(double *out, const double *in, const int64 *srci, const int64* srcj, 
              const double *srcv, int nsrc,const  int64 *nx_tensor, const int64 *ny_tensor){
  int64 NX, NY;
  cudaMemcpy(&NX, nx_tensor, sizeof(int64), cudaMemcpyDeviceToHost);
  cudaMemcpy(&NY, ny_tensor, sizeof(int64), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  Init<<<((NX+2)*(NY+2)+255)/256, 256>>>(out, in, (NX+2)*(NY+2));
  AddSource<<<(nsrc+255)/256, 256>>>(out, srci, srcj, srcv, nsrc, NY);
}

void backwardGPU(double *grad_src, double *grad_in, const double *grad_out, const double *in, const int64 *srci,
   const int64* srcj, const double *srcv, int nsrc, const int64 *nx_tensor, const int64 *ny_tensor){
  int64 NX, NY;
  cudaMemcpy(&NX, nx_tensor, sizeof(long long), cudaMemcpyDeviceToHost);
  cudaMemcpy(&NY, ny_tensor, sizeof(long long), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  Init<<<((NX+2)*(NY+2)+255)/256, 256>>>(grad_in, grad_out, (NX+2)*(NY+2));
  Zero<<<(nsrc+255)/256, 256>>>(grad_src, nsrc);
  AddSourceGrad<<<(nsrc+255)/256, 256>>>(grad_src, srci, srcj, grad_out, nsrc, NY);   
}