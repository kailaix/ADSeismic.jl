#include "cuda.h"

__global__ void ScatterAddOps_forward_kernel(double *out, const double *ipt, const int64 *ii,
    const double *update, int d, int n){
    int p =  blockIdx.x *blockDim.x + threadIdx.x;
    if (p < n){
        out[ii[p]-1] += update[p];
    }
 }
 
void Gpu_ScatterAddOps_forward(double *out, const double *ipt, const int64 *ii,
    const double *update, int d, int n){
    cudaMemcpy(out, ipt, sizeof(double) * d, cudaMemcpyDeviceToDevice);
    ScatterAddOps_forward_kernel<<< (n-1)/64 + 1, 64 >>>(out, ipt, ii, update, d, n);
 }


__global__ void ScatterAddOps_backward_kernel(double *grad_ipt, double *grad_update, 
    const double *grad_out,
      const double *out, const double *ipt, const int64 *ii,
     const double *update, int d, int n){
    
    int p =  blockIdx.x *blockDim.x + threadIdx.x;
    if (p < n){
        grad_update[p] = grad_out[ii[p]-1];
    }
}
 
 void Gpu_ScatterAddOps_backward(
   double *grad_ipt, double *grad_update, 
   const double *grad_out,
     const double *out, const double *ipt, const int64 *ii,
    const double *update, int d, int n){
    cudaMemcpy(grad_ipt, grad_out, sizeof(double)*d, cudaMemcpyDeviceToDevice);
    ScatterAddOps_backward_kernel<<< (n-1)/64, 64 >>>(grad_ipt, grad_update, grad_out, out, ipt, ii, 
                        update, d, n);
   }
 
 