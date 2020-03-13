#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"
#include<cmath>
#include<string> 
using std::string;
using namespace tensorflow;
// If you want to use the PyTorch feature, uncomment the following line
// #include "la.h" 

void forward(double *out, const double*vx, const double*vy, 
      const double*sigmaxx, const double*sigmayy, const double*sigmaxy, int64 nt,
      const int64 *rcvi, const int64 *rcvj, const int64 *rcvtype, int64 nrcv, int64 NX, int64 NY);

void backward(
    double*d_vx, double*d_vy, 
    double*d_sigmaxx, double*d_sigmayy, double*d_sigmaxy, const double *d_out, 
    int64 nt, const int64 *rcvi, const int64 *rcvj, const int64 *rcvtype, int64 nrcv, int64 NX, int64 NY);

void forwardGPU(double *out, const double*vx, const double*vy, 
      const double*sigmaxx, const double*sigmayy, const double*sigmaxy, long long nt,
      const long long *rcvi, const long long *rcvj, const long long *rcvtype, long long nrcv, const long long* nx, const long long* ny);

void backwardGPU(
    double*d_vx, double*d_vy, 
    double*d_sigmaxx, double*d_sigmayy, double*d_sigmaxy, const double *d_out, 
    long long  nt, const long long  *rcvi, const long long  *rcvj, const long long  *rcvtype, long long  nrcv, const long long * NX, const long long * NY);

REGISTER_OP("GetReceive")
.Input("vx : double")
  .Input("vy : double")
  .Input("sigmaxx : double")
  .Input("sigmayy : double")
  .Input("sigmaxy : double")
  .Input("rcvi : int64")
  .Input("rcvj : int64")
  .Input("rcvtype : int64")
  .Input("nx : int64")
  .Input("ny : int64")
  .Output("out : double")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    
        shape_inference::ShapeHandle vx_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &vx_shape));
        shape_inference::ShapeHandle vy_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &vy_shape));
        shape_inference::ShapeHandle sigmaxx_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 2, &sigmaxx_shape));
        shape_inference::ShapeHandle sigmayy_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 2, &sigmayy_shape));
        shape_inference::ShapeHandle sigmaxy_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(4), 2, &sigmaxy_shape));
        shape_inference::ShapeHandle rcvi_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(5), 1, &rcvi_shape));
        shape_inference::ShapeHandle rcvj_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(6), 1, &rcvj_shape));
        shape_inference::ShapeHandle rcvtype_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(7), 1, &rcvtype_shape));
        shape_inference::ShapeHandle nx_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(8), 0, &nx_shape));
        shape_inference::ShapeHandle ny_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(9), 0, &ny_shape));

        c->set_output(0, c->Matrix(c->Dim(c->input(6), 0),c->Dim(c->input(1), 0)));
    return Status::OK();
  });

REGISTER_OP("GetReceiveGrad")
  
  .Input("grad_out : double")
  .Input("out : double")
  .Input("vx : double")
  .Input("vy : double")
  .Input("sigmaxx : double")
  .Input("sigmayy : double")
  .Input("sigmaxy : double")
  .Input("rcvi : int64")
  .Input("rcvj : int64")
  .Input("rcvtype : int64")
  .Input("nx : int64")
  .Input("ny : int64")
  .Output("grad_vx : double")
  .Output("grad_vy : double")
  .Output("grad_sigmaxx : double")
  .Output("grad_sigmayy : double")
  .Output("grad_sigmaxy : double")
  .Output("grad_rcvi : int64")
  .Output("grad_rcvj : int64")
  .Output("grad_rcvtype : int64")
  .Output("grad_nx : int64")
  .Output("grad_ny : int64");

#if 1
class GetReceiveOp : public OpKernel {
private:
  
public:
  explicit GetReceiveOp(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {    
    DCHECK_EQ(10, context->num_inputs());
    
    
    const Tensor& vx = context->input(0);
    const Tensor& vy = context->input(1);
    const Tensor& sigmaxx = context->input(2);
    const Tensor& sigmayy = context->input(3);
    const Tensor& sigmaxy = context->input(4);
    const Tensor& rcvi = context->input(5);
    const Tensor& rcvj = context->input(6);
    const Tensor& rcvtype = context->input(7);
    const Tensor& nx = context->input(8);
    const Tensor& ny = context->input(9);
    
    
    const TensorShape& vx_shape = vx.shape();
    const TensorShape& vy_shape = vy.shape();
    const TensorShape& sigmaxx_shape = sigmaxx.shape();
    const TensorShape& sigmayy_shape = sigmayy.shape();
    const TensorShape& sigmaxy_shape = sigmaxy.shape();
    const TensorShape& rcvi_shape = rcvi.shape();
    const TensorShape& rcvj_shape = rcvj.shape();
    const TensorShape& rcvtype_shape = rcvtype.shape();
    const TensorShape& nx_shape = nx.shape();
    const TensorShape& ny_shape = ny.shape();
    
    
    DCHECK_EQ(vx_shape.dims(), 2);
    DCHECK_EQ(vy_shape.dims(), 2);
    DCHECK_EQ(sigmaxx_shape.dims(), 2);
    DCHECK_EQ(sigmayy_shape.dims(), 2);
    DCHECK_EQ(sigmaxy_shape.dims(), 2);
    DCHECK_EQ(rcvi_shape.dims(), 1);
    DCHECK_EQ(rcvj_shape.dims(), 1);
    DCHECK_EQ(rcvtype_shape.dims(), 1);
    DCHECK_EQ(nx_shape.dims(), 0);
    DCHECK_EQ(ny_shape.dims(), 0);

    // extra check
        
    // create output shape
    int nrcv = rcvi_shape.dim_size(0);
    int nt = sigmayy_shape.dim_size(0);

    TensorShape out_shape({nrcv,nt});
            
    // create output tensor
    
    Tensor* out = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &out));
    
    // get the corresponding Eigen tensors for data access
    
    auto vx_tensor = vx.flat<double>().data();
    auto vy_tensor = vy.flat<double>().data();
    auto sigmaxx_tensor = sigmaxx.flat<double>().data();
    auto sigmayy_tensor = sigmayy.flat<double>().data();
    auto sigmaxy_tensor = sigmaxy.flat<double>().data();
    auto rcvi_tensor = rcvi.flat<int64>().data();
    auto rcvj_tensor = rcvj.flat<int64>().data();
    auto rcvtype_tensor = rcvtype.flat<int64>().data();
    auto nx_tensor = nx.flat<int64>().data();
    auto ny_tensor = ny.flat<int64>().data();
    auto out_tensor = out->flat<double>().data();   

    // implement your forward function here 

    // TODO:
    forward(out_tensor, vx_tensor, vy_tensor, sigmaxx_tensor, sigmayy_tensor, sigmaxy_tensor,
        nt, rcvi_tensor, rcvj_tensor, rcvtype_tensor, nrcv, *nx_tensor, *ny_tensor);

  }
};
REGISTER_KERNEL_BUILDER(Name("GetReceive").Device(DEVICE_CPU), GetReceiveOp);

class GetReceiveGradOp : public OpKernel {
private:
  
public:
  explicit GetReceiveGradOp(OpKernelConstruction* context) : OpKernel(context) {
    
  }
  
  void Compute(OpKernelContext* context) override {
    
    
    const Tensor& grad_out = context->input(0);
    const Tensor& out = context->input(1);
    const Tensor& vx = context->input(2);
    const Tensor& vy = context->input(3);
    const Tensor& sigmaxx = context->input(4);
    const Tensor& sigmayy = context->input(5);
    const Tensor& sigmaxy = context->input(6);
    const Tensor& rcvi = context->input(7);
    const Tensor& rcvj = context->input(8);
    const Tensor& rcvtype = context->input(9);
    const Tensor& nx = context->input(10);
    const Tensor& ny = context->input(11);
    
    
    const TensorShape& grad_out_shape = grad_out.shape();
    const TensorShape& out_shape = out.shape();
    const TensorShape& vx_shape = vx.shape();
    const TensorShape& vy_shape = vy.shape();
    const TensorShape& sigmaxx_shape = sigmaxx.shape();
    const TensorShape& sigmayy_shape = sigmayy.shape();
    const TensorShape& sigmaxy_shape = sigmaxy.shape();
    const TensorShape& rcvi_shape = rcvi.shape();
    const TensorShape& rcvj_shape = rcvj.shape();
    const TensorShape& rcvtype_shape = rcvtype.shape();
    const TensorShape& nx_shape = nx.shape();
    const TensorShape& ny_shape = ny.shape();
    
    
    DCHECK_EQ(grad_out_shape.dims(), 2);
    DCHECK_EQ(out_shape.dims(), 2);
    DCHECK_EQ(vx_shape.dims(), 2);
    DCHECK_EQ(vy_shape.dims(), 2);
    DCHECK_EQ(sigmaxx_shape.dims(), 2);
    DCHECK_EQ(sigmayy_shape.dims(), 2);
    DCHECK_EQ(sigmaxy_shape.dims(), 2);
    DCHECK_EQ(rcvi_shape.dims(), 1);
    DCHECK_EQ(rcvj_shape.dims(), 1);
    DCHECK_EQ(rcvtype_shape.dims(), 1);
    DCHECK_EQ(nx_shape.dims(), 0);
    DCHECK_EQ(ny_shape.dims(), 0);

    // extra check
    // int m = Example.dim_size(0);
        
    // create output shape
    
    TensorShape grad_vx_shape(vx_shape);
    TensorShape grad_vy_shape(vy_shape);
    TensorShape grad_sigmaxx_shape(sigmaxx_shape);
    TensorShape grad_sigmayy_shape(sigmayy_shape);
    TensorShape grad_sigmaxy_shape(sigmaxy_shape);
    TensorShape grad_rcvi_shape(rcvi_shape);
    TensorShape grad_rcvj_shape(rcvj_shape);
    TensorShape grad_rcvtype_shape(rcvtype_shape);
    TensorShape grad_nx_shape(nx_shape);
    TensorShape grad_ny_shape(ny_shape);
            
    // create output tensor
    
    Tensor* grad_vx = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_vx_shape, &grad_vx));
    Tensor* grad_vy = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, grad_vy_shape, &grad_vy));
    Tensor* grad_sigmaxx = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(2, grad_sigmaxx_shape, &grad_sigmaxx));
    Tensor* grad_sigmayy = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(3, grad_sigmayy_shape, &grad_sigmayy));
    Tensor* grad_sigmaxy = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(4, grad_sigmaxy_shape, &grad_sigmaxy));
    Tensor* grad_rcvi = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(5, grad_rcvi_shape, &grad_rcvi));
    Tensor* grad_rcvj = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(6, grad_rcvj_shape, &grad_rcvj));
    Tensor* grad_rcvtype = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(7, grad_rcvtype_shape, &grad_rcvtype));
    Tensor* grad_nx = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(8, grad_nx_shape, &grad_nx));
    Tensor* grad_ny = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(9, grad_ny_shape, &grad_ny));
    
    // get the corresponding Eigen tensors for data access
    
    auto vx_tensor = vx.flat<double>().data();
    auto vy_tensor = vy.flat<double>().data();
    auto sigmaxx_tensor = sigmaxx.flat<double>().data();
    auto sigmayy_tensor = sigmayy.flat<double>().data();
    auto sigmaxy_tensor = sigmaxy.flat<double>().data();
    auto rcvi_tensor = rcvi.flat<int64>().data();
    auto rcvj_tensor = rcvj.flat<int64>().data();
    auto rcvtype_tensor = rcvtype.flat<int64>().data();
    auto nx_tensor = nx.flat<int64>().data();
    auto ny_tensor = ny.flat<int64>().data();
    auto grad_out_tensor = grad_out.flat<double>().data();
    auto out_tensor = out.flat<double>().data();
    auto grad_vx_tensor = grad_vx->flat<double>().data();
    auto grad_vy_tensor = grad_vy->flat<double>().data();
    auto grad_sigmaxx_tensor = grad_sigmaxx->flat<double>().data();
    auto grad_sigmayy_tensor = grad_sigmayy->flat<double>().data();
    auto grad_sigmaxy_tensor = grad_sigmaxy->flat<double>().data();
    auto grad_rcvi_tensor = grad_rcvi->flat<int64>().data();
    auto grad_rcvj_tensor = grad_rcvj->flat<int64>().data();
    auto grad_rcvtype_tensor = grad_rcvtype->flat<int64>().data();
    auto grad_nx_tensor = grad_nx->flat<int64>().data();
    auto grad_ny_tensor = grad_ny->flat<int64>().data();   

    // implement your backward function here 
    int nrcv = rcvi_shape.dim_size(0);
    int nt = sigmayy_shape.dim_size(0);
    // TODO:
    backward(grad_vx_tensor, grad_vy_tensor, grad_sigmaxx_tensor, grad_sigmayy_tensor, grad_sigmaxy_tensor,
        grad_out_tensor, nt, rcvi_tensor, rcvj_tensor, rcvtype_tensor, nrcv, *nx_tensor, *ny_tensor);
  }
};
REGISTER_KERNEL_BUILDER(Name("GetReceiveGrad").Device(DEVICE_CPU), GetReceiveGradOp);

#endif


#ifndef NOGPU

class GetReceiveOpGPU : public OpKernel {
private:
  
public:
  explicit GetReceiveOpGPU(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {    
    DCHECK_EQ(10, context->num_inputs());
    
    
    const Tensor& vx = context->input(0);
    const Tensor& vy = context->input(1);
    const Tensor& sigmaxx = context->input(2);
    const Tensor& sigmayy = context->input(3);
    const Tensor& sigmaxy = context->input(4);
    const Tensor& rcvi = context->input(5);
    const Tensor& rcvj = context->input(6);
    const Tensor& rcvtype = context->input(7);
    const Tensor& nx = context->input(8);
    const Tensor& ny = context->input(9);
    
    
    const TensorShape& vx_shape = vx.shape();
    const TensorShape& vy_shape = vy.shape();
    const TensorShape& sigmaxx_shape = sigmaxx.shape();
    const TensorShape& sigmayy_shape = sigmayy.shape();
    const TensorShape& sigmaxy_shape = sigmaxy.shape();
    const TensorShape& rcvi_shape = rcvi.shape();
    const TensorShape& rcvj_shape = rcvj.shape();
    const TensorShape& rcvtype_shape = rcvtype.shape();
    const TensorShape& nx_shape = nx.shape();
    const TensorShape& ny_shape = ny.shape();
    
    
    DCHECK_EQ(vx_shape.dims(), 2);
    DCHECK_EQ(vy_shape.dims(), 2);
    DCHECK_EQ(sigmaxx_shape.dims(), 2);
    DCHECK_EQ(sigmayy_shape.dims(), 2);
    DCHECK_EQ(sigmaxy_shape.dims(), 2);
    DCHECK_EQ(rcvi_shape.dims(), 1);
    DCHECK_EQ(rcvj_shape.dims(), 1);
    DCHECK_EQ(rcvtype_shape.dims(), 1);
    DCHECK_EQ(nx_shape.dims(), 0);
    DCHECK_EQ(ny_shape.dims(), 0);

    // extra check
        
    // create output shape
    int nrcv = rcvi_shape.dim_size(0);
    int nt = sigmayy_shape.dim_size(0);

    TensorShape out_shape({nrcv,nt});
            
    // create output tensor
    
    Tensor* out = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &out));
    
    // get the corresponding Eigen tensors for data access
    
    auto vx_tensor = vx.flat<double>().data();
    auto vy_tensor = vy.flat<double>().data();
    auto sigmaxx_tensor = sigmaxx.flat<double>().data();
    auto sigmayy_tensor = sigmayy.flat<double>().data();
    auto sigmaxy_tensor = sigmaxy.flat<double>().data();
    auto rcvi_tensor = rcvi.flat<int64>().data();
    auto rcvj_tensor = rcvj.flat<int64>().data();
    auto rcvtype_tensor = rcvtype.flat<int64>().data();
    auto nx_tensor = nx.flat<int64>().data();
    auto ny_tensor = ny.flat<int64>().data();
    auto out_tensor = out->flat<double>().data();   

    // implement your forward function here 

    forwardGPU(out_tensor, vx_tensor, vy_tensor, 
      sigmaxx_tensor, sigmayy_tensor, sigmaxy_tensor, nt,
      rcvi_tensor, rcvj_tensor, rcvtype_tensor, nrcv, nx_tensor, ny_tensor);


  }
};
REGISTER_KERNEL_BUILDER(Name("GetReceive").Device(DEVICE_GPU), GetReceiveOpGPU);




class GetReceiveGradOpGPU : public OpKernel {
private:
  
public:
  explicit GetReceiveGradOpGPU(OpKernelConstruction* context) : OpKernel(context) {
    
  }
  
  void Compute(OpKernelContext* context) override {
    
    
    const Tensor& grad_out = context->input(0);
    const Tensor& out = context->input(1);
    const Tensor& vx = context->input(2);
    const Tensor& vy = context->input(3);
    const Tensor& sigmaxx = context->input(4);
    const Tensor& sigmayy = context->input(5);
    const Tensor& sigmaxy = context->input(6);
    const Tensor& rcvi = context->input(7);
    const Tensor& rcvj = context->input(8);
    const Tensor& rcvtype = context->input(9);
    const Tensor& nx = context->input(10);
    const Tensor& ny = context->input(11);
    
    
    const TensorShape& grad_out_shape = grad_out.shape();
    const TensorShape& out_shape = out.shape();
    const TensorShape& vx_shape = vx.shape();
    const TensorShape& vy_shape = vy.shape();
    const TensorShape& sigmaxx_shape = sigmaxx.shape();
    const TensorShape& sigmayy_shape = sigmayy.shape();
    const TensorShape& sigmaxy_shape = sigmaxy.shape();
    const TensorShape& rcvi_shape = rcvi.shape();
    const TensorShape& rcvj_shape = rcvj.shape();
    const TensorShape& rcvtype_shape = rcvtype.shape();
    const TensorShape& nx_shape = nx.shape();
    const TensorShape& ny_shape = ny.shape();
    
    
    DCHECK_EQ(grad_out_shape.dims(), 2);
    DCHECK_EQ(out_shape.dims(), 2);
    DCHECK_EQ(vx_shape.dims(), 2);
    DCHECK_EQ(vy_shape.dims(), 2);
    DCHECK_EQ(sigmaxx_shape.dims(), 2);
    DCHECK_EQ(sigmayy_shape.dims(), 2);
    DCHECK_EQ(sigmaxy_shape.dims(), 2);
    DCHECK_EQ(rcvi_shape.dims(), 1);
    DCHECK_EQ(rcvj_shape.dims(), 1);
    DCHECK_EQ(rcvtype_shape.dims(), 1);
    DCHECK_EQ(nx_shape.dims(), 0);
    DCHECK_EQ(ny_shape.dims(), 0);

    // extra check
    // int m = Example.dim_size(0);
        
    // create output shape
    
    TensorShape grad_vx_shape(vx_shape);
    TensorShape grad_vy_shape(vy_shape);
    TensorShape grad_sigmaxx_shape(sigmaxx_shape);
    TensorShape grad_sigmayy_shape(sigmayy_shape);
    TensorShape grad_sigmaxy_shape(sigmaxy_shape);
    TensorShape grad_rcvi_shape(rcvi_shape);
    TensorShape grad_rcvj_shape(rcvj_shape);
    TensorShape grad_rcvtype_shape(rcvtype_shape);
    TensorShape grad_nx_shape(nx_shape);
    TensorShape grad_ny_shape(ny_shape);
            
    // create output tensor
    
    Tensor* grad_vx = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_vx_shape, &grad_vx));
    Tensor* grad_vy = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, grad_vy_shape, &grad_vy));
    Tensor* grad_sigmaxx = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(2, grad_sigmaxx_shape, &grad_sigmaxx));
    Tensor* grad_sigmayy = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(3, grad_sigmayy_shape, &grad_sigmayy));
    Tensor* grad_sigmaxy = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(4, grad_sigmaxy_shape, &grad_sigmaxy));
    Tensor* grad_rcvi = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(5, grad_rcvi_shape, &grad_rcvi));
    Tensor* grad_rcvj = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(6, grad_rcvj_shape, &grad_rcvj));
    Tensor* grad_rcvtype = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(7, grad_rcvtype_shape, &grad_rcvtype));
    Tensor* grad_nx = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(8, grad_nx_shape, &grad_nx));
    Tensor* grad_ny = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(9, grad_ny_shape, &grad_ny));
    
    // get the corresponding Eigen tensors for data access
    
    auto vx_tensor = vx.flat<double>().data();
    auto vy_tensor = vy.flat<double>().data();
    auto sigmaxx_tensor = sigmaxx.flat<double>().data();
    auto sigmayy_tensor = sigmayy.flat<double>().data();
    auto sigmaxy_tensor = sigmaxy.flat<double>().data();
    auto rcvi_tensor = rcvi.flat<int64>().data();
    auto rcvj_tensor = rcvj.flat<int64>().data();
    auto rcvtype_tensor = rcvtype.flat<int64>().data();
    auto nx_tensor = nx.flat<int64>().data();
    auto ny_tensor = ny.flat<int64>().data();
    auto grad_out_tensor = grad_out.flat<double>().data();
    auto out_tensor = out.flat<double>().data();
    auto grad_vx_tensor = grad_vx->flat<double>().data();
    auto grad_vy_tensor = grad_vy->flat<double>().data();
    auto grad_sigmaxx_tensor = grad_sigmaxx->flat<double>().data();
    auto grad_sigmayy_tensor = grad_sigmayy->flat<double>().data();
    auto grad_sigmaxy_tensor = grad_sigmaxy->flat<double>().data();
    auto grad_rcvi_tensor = grad_rcvi->flat<int64>().data();
    auto grad_rcvj_tensor = grad_rcvj->flat<int64>().data();
    auto grad_rcvtype_tensor = grad_rcvtype->flat<int64>().data();
    auto grad_nx_tensor = grad_nx->flat<int64>().data();
    auto grad_ny_tensor = grad_ny->flat<int64>().data();   

    // implement your backward function here 
    int nrcv = rcvi_shape.dim_size(0);
    int nt = sigmayy_shape.dim_size(0);
    // TODO:
    backwardGPU(grad_vx_tensor, grad_vy_tensor, grad_sigmaxx_tensor, grad_sigmayy_tensor, grad_sigmaxy_tensor,
        grad_out_tensor, nt, rcvi_tensor, rcvj_tensor, rcvtype_tensor, nrcv, nx_tensor, ny_tensor);
  }
};
REGISTER_KERNEL_BUILDER(Name("GetReceiveGrad").Device(DEVICE_GPU), GetReceiveGradOpGPU);

#endif