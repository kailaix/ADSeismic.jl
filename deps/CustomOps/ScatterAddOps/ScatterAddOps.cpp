#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"
#include<cmath>

// Signatures for GPU kernels here 


using namespace tensorflow;
#include "ScatterAddOps.h"


REGISTER_OP("ScatterAddOps")
.Input("ipt : double")
.Input("ii : int64")
.Input("vv : double")
.Output("out : double")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    
        shape_inference::ShapeHandle ipt_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &ipt_shape));
        shape_inference::ShapeHandle ii_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &ii_shape));
        shape_inference::ShapeHandle vv_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &vv_shape));

        c->set_output(0, c->Vector(-1));
    return Status::OK();
  });

REGISTER_OP("ScatterAddOpsGrad")
.Input("grad_out : double")
.Input("out : double")
.Input("ipt : double")
.Input("ii : int64")
.Input("vv : double")
.Output("grad_ipt : double")
.Output("grad_ii : int64")
.Output("grad_vv : double");

/*-------------------------------------------------------------------------------------*/

class ScatterAddOpsOp : public OpKernel {
private:
  
public:
  explicit ScatterAddOpsOp(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {    
    DCHECK_EQ(3, context->num_inputs());
    
    
    const Tensor& ipt = context->input(0);
    const Tensor& ii = context->input(1);
    const Tensor& vv = context->input(2);
    
    
    const TensorShape& ipt_shape = ipt.shape();
    const TensorShape& ii_shape = ii.shape();
    const TensorShape& vv_shape = vv.shape();
    
    
    DCHECK_EQ(ipt_shape.dims(), 1);
    DCHECK_EQ(ii_shape.dims(), 1);
    DCHECK_EQ(vv_shape.dims(), 1);

    // extra check
        
    // create output shape
    int d = ipt_shape.dim_size(0), n = vv_shape.dim_size(0);
    TensorShape out_shape({d});
            
    // create output tensor
    
    Tensor* out = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &out));
    
    // get the corresponding Eigen tensors for data access
    
    auto ipt_tensor = ipt.flat<double>().data();
    auto ii_tensor = ii.flat<int64>().data();
    auto vv_tensor = vv.flat<double>().data();
    auto out_tensor = out->flat<double>().data();   

    // implement your forward function here 

    // TODO:
     ScatterAddOps_forward(out_tensor, ipt_tensor, ii_tensor, vv_tensor, d, n);
       

  }
};
REGISTER_KERNEL_BUILDER(Name("ScatterAddOps").Device(DEVICE_CPU), ScatterAddOpsOp);



class ScatterAddOpsGradOp : public OpKernel {
private:
  
public:
  explicit ScatterAddOpsGradOp(OpKernelConstruction* context) : OpKernel(context) {
    
  }
  
  void Compute(OpKernelContext* context) override {
    
    
    const Tensor& grad_out = context->input(0);
    const Tensor& out = context->input(1);
    const Tensor& ipt = context->input(2);
    const Tensor& ii = context->input(3);
    const Tensor& vv = context->input(4);
    
    
    const TensorShape& grad_out_shape = grad_out.shape();
    const TensorShape& out_shape = out.shape();
    const TensorShape& ipt_shape = ipt.shape();
    const TensorShape& ii_shape = ii.shape();
    const TensorShape& vv_shape = vv.shape();
    
    
    DCHECK_EQ(grad_out_shape.dims(), 1);
    DCHECK_EQ(out_shape.dims(), 1);
    DCHECK_EQ(ipt_shape.dims(), 1);
    DCHECK_EQ(ii_shape.dims(), 1);
    DCHECK_EQ(vv_shape.dims(), 1);

    // extra check
    // int m = Example.dim_size(0);
        
    // create output shape
    
    TensorShape grad_ipt_shape(ipt_shape);
    TensorShape grad_ii_shape(ii_shape);
    TensorShape grad_vv_shape(vv_shape);
            
    // create output tensor
    
    Tensor* grad_ipt = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_ipt_shape, &grad_ipt));
    Tensor* grad_ii = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, grad_ii_shape, &grad_ii));
    Tensor* grad_vv = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(2, grad_vv_shape, &grad_vv));
    
    // get the corresponding Eigen tensors for data access
    
    auto ipt_tensor = ipt.flat<double>().data();
    auto ii_tensor = ii.flat<int64>().data();
    auto vv_tensor = vv.flat<double>().data();
    auto grad_out_tensor = grad_out.flat<double>().data();
    auto out_tensor = out.flat<double>().data();
    auto grad_ipt_tensor = grad_ipt->flat<double>().data();
    auto grad_vv_tensor = grad_vv->flat<double>().data();   

    // implement your backward function here 

    // TODO:
    int d = ipt_shape.dim_size(0), n = vv_shape.dim_size(0);
    grad_ipt->flat<double>().setZero();
    ScatterAddOps_backward(
      grad_ipt_tensor, grad_vv_tensor, grad_out_tensor,
      out_tensor, ipt_tensor, ii_tensor, vv_tensor, d, n);
    
  }
};
REGISTER_KERNEL_BUILDER(Name("ScatterAddOpsGrad").Device(DEVICE_CPU), ScatterAddOpsGradOp);



/***************************************************************************************
**********************            GPU Operator            ******************************
***************************************************************************************/

void Gpu_ScatterAddOps_forward(double *out, const double *ipt, const long long *ii,
    const double *update, int d, int n);
void Gpu_ScatterAddOps_backward(
   double *grad_ipt, double *grad_update, 
   const double *grad_out,
     const double *out, const double *ipt, const long long *ii,
    const double *update, int d, int n);

#ifdef GOOGLE_CUDA
class ScatterAddOpsOpGPU : public OpKernel {
private:
  
public:
  explicit ScatterAddOpsOpGPU(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {    
    DCHECK_EQ(3, context->num_inputs());
    
    
    const Tensor& ipt = context->input(0);
    const Tensor& ii = context->input(1);
    const Tensor& vv = context->input(2);
    
    
    const TensorShape& ipt_shape = ipt.shape();
    const TensorShape& ii_shape = ii.shape();
    const TensorShape& vv_shape = vv.shape();
    
    
    DCHECK_EQ(ipt_shape.dims(), 1);
    DCHECK_EQ(ii_shape.dims(), 1);
    DCHECK_EQ(vv_shape.dims(), 1);

    // extra check
        
    // create output shape
    int d = ipt_shape.dim_size(0), n = vv_shape.dim_size(0);
    TensorShape out_shape({d});
            
    // create output tensor
    
    Tensor* out = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &out));
    
    // get the corresponding Eigen tensors for data access
    
    auto ipt_tensor = ipt.flat<double>().data();
    auto ii_tensor = ii.flat<int64>().data();
    auto vv_tensor = vv.flat<double>().data();
    auto out_tensor = out->flat<double>().data();   

    // implement your forward function here 

    // TODO:
    Gpu_ScatterAddOps_forward(out_tensor, ipt_tensor, ii_tensor, vv_tensor, d, n);

  }
};
REGISTER_KERNEL_BUILDER(Name("ScatterAddOps").Device(DEVICE_GPU), ScatterAddOpsOpGPU);

class ScatterAddOpsGradOpGPU : public OpKernel {
private:
  
public:
  explicit ScatterAddOpsGradOpGPU(OpKernelConstruction* context) : OpKernel(context) {
    
  }
  
  void Compute(OpKernelContext* context) override {
    
    
    const Tensor& grad_out = context->input(0);
    const Tensor& out = context->input(1);
    const Tensor& ipt = context->input(2);
    const Tensor& ii = context->input(3);
    const Tensor& vv = context->input(4);
    
    
    const TensorShape& grad_out_shape = grad_out.shape();
    const TensorShape& out_shape = out.shape();
    const TensorShape& ipt_shape = ipt.shape();
    const TensorShape& ii_shape = ii.shape();
    const TensorShape& vv_shape = vv.shape();
    
    
    DCHECK_EQ(grad_out_shape.dims(), 1);
    DCHECK_EQ(out_shape.dims(), 1);
    DCHECK_EQ(ipt_shape.dims(), 1);
    DCHECK_EQ(ii_shape.dims(), 1);
    DCHECK_EQ(vv_shape.dims(), 1);

    // extra check
    // int m = Example.dim_size(0);
        
    // create output shape
    
    TensorShape grad_ipt_shape(ipt_shape);
    TensorShape grad_ii_shape(ii_shape);
    TensorShape grad_vv_shape(vv_shape);
            
    // create output tensor
    
    Tensor* grad_ipt = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_ipt_shape, &grad_ipt));
    Tensor* grad_ii = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, grad_ii_shape, &grad_ii));
    Tensor* grad_vv = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(2, grad_vv_shape, &grad_vv));
    
    // get the corresponding Eigen tensors for data access
    
    auto ipt_tensor = ipt.flat<double>().data();
    auto ii_tensor = ii.flat<int64>().data();
    auto vv_tensor = vv.flat<double>().data();
    auto grad_out_tensor = grad_out.flat<double>().data();
    auto out_tensor = out.flat<double>().data();
    auto grad_ipt_tensor = grad_ipt->flat<double>().data();
    auto grad_vv_tensor = grad_vv->flat<double>().data();   

    // implement your backward function here 

    // TODO:
    int d = ipt_shape.dim_size(0), n = vv_shape.dim_size(0);
    Gpu_ScatterAddOps_backward(
      grad_ipt_tensor, grad_vv_tensor, grad_out_tensor,
      out_tensor, ipt_tensor, ii_tensor, vv_tensor, d, n);
    
  }
};
REGISTER_KERNEL_BUILDER(Name("ScatterAddOpsGrad").Device(DEVICE_GPU), ScatterAddOpsGradOpGPU);

#endif
