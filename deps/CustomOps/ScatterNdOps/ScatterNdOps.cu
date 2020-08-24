#define GOOGLE_CUDA 1
#define EIGEN_USE_GPU

#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"

__global__ void ScatterNdOps_forward_kernel(double *out, const int64*ii, const double *update, int n){
    int p =  blockIdx.x *blockDim.x + threadIdx.x;
    if (p<n){
        out[ii[i]-1] = update[i];
    }
}

void Gpu_ScatterNdOps_forward(double *out, const int64 *ii,
    const double *update, int n){
    ScatterNdOps_forward_kernel<<< (n - 1)/64 + 1, 64 >>>(out, ii, update, n);
 }

 
 __global__ void ScatterNdOps_backward_kernel(double *grad_update, 
    const double *grad_out,
    const double *out, const int64 *ii,
    const double *update, int n){
    int p =  blockIdx.x *blockDim.x + threadIdx.x;
    if (p<n) {
        grad_update[i] = grad_out[ii[i]-1];
    }
}

void Gpu_ScatterNdOps_backward(
    double *grad_update, 
    const double *grad_out,
    const double *out, const int64 *ii,
    const double *update, int n){
    ScatterNdOps_backward_kernel<<< (n - 1)/64 + 1, 64 >>>(grad_update, grad_out, out, ii, update, n);
 }

 int get_ScatterNdOps_num(int64 *out, const int64 *m){
    cudaMemcpy(out, m, sizeof(int64), cudaMemcpyHostToDevice);
 }