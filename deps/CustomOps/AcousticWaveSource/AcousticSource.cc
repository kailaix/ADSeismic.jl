#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"
#include<cmath>
#include<string> 
#include<cmath>
#include<string> 
using std::string;
using namespace tensorflow;
void forward(double *out, const double *in, const int64 *srci, const int64* srcj, const double *srcv, int nsrc, int64 NX, int64 NY){
  for(int i=0;i<(NX+2)*(NY+2);i++) out[i] = in[i];
  for(int i=0;i<nsrc;i++){
      int idx = (srci[i]-1)*(NY+2)+srcj[i]-1;
      out[idx] += srcv[i];
  }    
}

void backward(double *grad_src, double *grad_in, const double *grad_out, const double *in, const int64 *srci, const int64* srcj, const double *srcv, int nsrc, int64 NX, int64 NY){
  for(int i=0;i<(NX+2)*(NY+2);i++) grad_in[i] = grad_out[i];
  for(int i=0;i<nsrc;i++){
      int idx = (srci[i]-1)*(NY+2)+srcj[i]-1;
      grad_src[i] = grad_out[idx];
  }    
}