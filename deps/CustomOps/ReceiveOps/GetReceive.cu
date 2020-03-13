#include "cuda.h"
typedef  long long int64;

__global__ void ReceiveFun(double *out, const double*vx, const double*vy, 
  const double*sigmaxx, const double*sigmayy, const double*sigmaxy, int64 nt,
  const int64 *rcvi, const int64 *rcvj, const int64 *rcvtype, int64 nrcv, int64 NX, int64 NY){
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i>=nrcv) return;

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

void forwardGPU(double *out, const double*vx, const double*vy, 
      const double*sigmaxx, const double*sigmayy, const double*sigmaxy, int64 nt,
      const int64 *rcvi, const int64 *rcvj, const int64 *rcvtype, int64 nrcv, const int64* nx, const int64* ny){
    long long NX, NY;
    cudaMemcpy(&NX, nx, sizeof(long long), cudaMemcpyDeviceToHost);
    cudaMemcpy(&NY, ny, sizeof(long long), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    ReceiveFun<<<(nrcv+255)/256, 256>>>(out, vx, vy, sigmaxx, sigmayy, sigmaxy, nt,
      rcvi, rcvj, rcvtype, nrcv, NX, NY);
}


__global__ void Zero(const long long size, double* out) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if(i<size) out[i] = 0.0;
}

__global__ void ReceiveGrad(
  double*d_vx, double*d_vy, 
    double*d_sigmaxx, double*d_sigmayy, double*d_sigmaxy, const double *d_out, 
    int64 nt, const int64 *rcvi, const int64 *rcvj, const int64 *rcvtype, int64 nrcv, int64 NX, int64 NY) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if(i>=nrcv) return;
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

void backwardGPU(
    double*d_vx, double*d_vy, 
    double*d_sigmaxx, double*d_sigmayy, double*d_sigmaxy, const double *d_out, 
    int64 nt, const int64 *rcvi, const int64 *rcvj, const int64 *rcvtype, int64 nrcv, const int64* nx, const int64* ny){
  long long NX, NY;
  cudaMemcpy(&NX, nx, sizeof(long long), cudaMemcpyDeviceToHost);
  cudaMemcpy(&NY, ny, sizeof(long long), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  Zero<<<(nt*(NX+2)*(NY+2)+255)/256, 256>>>(nt*(NX+2)*(NY+2), d_vx);
  Zero<<<(nt*(NX+2)*(NY+2)+255)/256, 256>>>(nt*(NX+2)*(NY+2), d_vy);
  Zero<<<(nt*(NX+2)*(NY+2)+255)/256, 256>>>(nt*(NX+2)*(NY+2), d_sigmaxx);
  Zero<<<(nt*(NX+2)*(NY+2)+255)/256, 256>>>(nt*(NX+2)*(NY+2), d_sigmayy);
  Zero<<<(nt*(NX+2)*(NY+2)+255)/256, 256>>>(nt*(NX+2)*(NY+2), d_sigmaxy);
  ReceiveGrad<<<(nrcv+255)/256, 256>>>(d_vx, d_vy, d_sigmaxx, d_sigmayy, d_sigmaxy,
        d_out, nt, rcvi, rcvj, rcvtype, nrcv, NX, NY);
}