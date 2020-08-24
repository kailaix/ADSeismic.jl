#include "cuda.h"
__global__ void GatherOps_forward_kernel(double *out, const double *v, const long long *ii, int n){
    int p =  blockIdx.x *blockDim.x + threadIdx.x;
    if (p < n){
        out[p] = v[ii[p]-1];
    }
}

void Gpu_GatherOps_forward(double *out, const double *v, const long long *ii, int n){
    GatherOps_forward_kernel<<< (n-1)/64 + 1, 64 >>>(out, v, ii, n);
  }

__global__ void GatherOps_backward_kernel(double *grad_v, 
    const double *grad_out, 
    const double *out, const double *v, const long long *ii, int n){
    int p =  blockIdx.x *blockDim.x + threadIdx.x;
    if (p < n){
        grad_v[ii[p]-1] = grad_out[p];
    }
}
  
void Gpu_GatherOps_backward(
double *grad_v, 
const double *grad_out, 
const double *out, const double *v, const long long *ii, int n
){
    GatherOps_backward_kernel<<< (n-1)/64 + 1, 64 >>>(grad_v, grad_out, out, v, ii, n);
}