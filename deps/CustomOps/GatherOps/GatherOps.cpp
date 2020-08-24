#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"
#include<cmath>

// Signatures for GPU kernels here 


using namespace tensorflow;
#include "GatherOps.h"


REGISTER_OP("GatherOps")
.Input("ipt : double")
.Input("ii : int64")
.Output("out : double")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    
        shape_inference::ShapeHandle ipt_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &ipt_shape));
        shape_inference::ShapeHandle ii_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &ii_shape));

        c->set_output(0, c->Vector(-1));
    return Status::OK();
  });

REGISTER_OP("GatherOpsGrad")
.Input("grad_out : double")
.Input("out : double")
.Input("ipt : double")
.Input("ii : int64")
.Output("grad_ipt : double")
.Output("grad_ii : int64");

/*-------------------------------------------------------------------------------------*/

class GatherOpsOp : public OpKernel {
private:
  
public:
  explicit GatherOpsOp(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {    
    DCHECK_EQ(2, context->num_inputs());
    
    
    const Tensor& ipt = context->input(0);
    const Tensor& ii = context->input(1);
    
    
    const TensorShape& ipt_shape = ipt.shape();
    const TensorShape& ii_shape = ii.shape();
    
    
    DCHECK_EQ(ipt_shape.dims(), 1);
    DCHECK_EQ(ii_shape.dims(), 1);

    // extra check
        
    // create output shape
    int n = ii_shape.dim_size(0);
    TensorShape out_shape({n});
            
    // create output tensor
    
    Tensor* out = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &out));
    
    // get the corresponding Eigen tensors for data access
    
    auto ipt_tensor = ipt.flat<double>().data();
    auto ii_tensor = ii.flat<int64>().data();
    auto out_tensor = out->flat<double>().data();   

    // implement your forward function here 

    // TODO:
    GatherOps_forward(
      out_tensor, ipt_tensor, ii_tensor, n);

  }
};
REGISTER_KERNEL_BUILDER(Name("GatherOps").Device(DEVICE_CPU), GatherOpsOp);



class GatherOpsGradOp : public OpKernel {
private:
  
public:
  explicit GatherOpsGradOp(OpKernelConstruction* context) : OpKernel(context) {
    
  }
  
  void Compute(OpKernelContext* context) override {
    
    
    const Tensor& grad_out = context->input(0);
    const Tensor& out = context->input(1);
    const Tensor& ipt = context->input(2);
    const Tensor& ii = context->input(3);
    
    
    const TensorShape& grad_out_shape = grad_out.shape();
    const TensorShape& out_shape = out.shape();
    const TensorShape& ipt_shape = ipt.shape();
    const TensorShape& ii_shape = ii.shape();
    
    
    DCHECK_EQ(grad_out_shape.dims(), 1);
    DCHECK_EQ(out_shape.dims(), 1);
    DCHECK_EQ(ipt_shape.dims(), 1);
    DCHECK_EQ(ii_shape.dims(), 1);

    // extra check
    // int m = Example.dim_size(0);
        
    // create output shape
    
    TensorShape grad_ipt_shape(ipt_shape);
    TensorShape grad_ii_shape(ii_shape);
            
    // create output tensor
    
    Tensor* grad_ipt = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_ipt_shape, &grad_ipt));
    Tensor* grad_ii = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, grad_ii_shape, &grad_ii));
    
    // get the corresponding Eigen tensors for data access
    
    auto ipt_tensor = ipt.flat<double>().data();
    auto ii_tensor = ii.flat<int64>().data();
    auto grad_out_tensor = grad_out.flat<double>().data();
    auto out_tensor = out.flat<double>().data();
    auto grad_ipt_tensor = grad_ipt->flat<double>().data();   

    // implement your backward function here 

    // TODO:
    int n = ii_shape.dim_size(0);
    grad_ipt->flat<double>().setZero();
  GatherOps_backward(
      grad_ipt_tensor, grad_out_tensor, 
      out_tensor, ipt_tensor, ii_tensor, n);
    
  }
};
REGISTER_KERNEL_BUILDER(Name("GatherOpsGrad").Device(DEVICE_CPU), GatherOpsGradOp);


/***************************************************************************************
**********************            GPU Operator            ******************************
***************************************************************************************/


#ifdef GOOGLE_CUDA
class GatherOpsOpGPU : public OpKernel {
private:
  
public:
  explicit GatherOpsOpGPU(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {    
    DCHECK_EQ(2, context->num_inputs());
    
    
    const Tensor& ipt = context->input(0);
    const Tensor& ii = context->input(1);
    
    
    const TensorShape& ipt_shape = ipt.shape();
    const TensorShape& ii_shape = ii.shape();
    
    
    DCHECK_EQ(ipt_shape.dims(), 1);
    DCHECK_EQ(ii_shape.dims(), 1);

    // extra check
        
    // create output shape
    
    TensorShape out_shape({-1});
            
    // create output tensor
    
    Tensor* out = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &out));
    
    // get the corresponding Eigen tensors for data access
    
    auto ipt_tensor = ipt.flat<double>().data();
    auto ii_tensor = ii.flat<int64>().data();
    auto out_tensor = out->flat<double>().data();   

    // implement your forward function here 

    // TODO:

  }
};
REGISTER_KERNEL_BUILDER(Name("GatherOps").Device(DEVICE_GPU), GatherOpsOpGPU);

class GatherOpsGradOpGPU : public OpKernel {
private:
  
public:
  explicit GatherOpsGradOpGPU(OpKernelConstruction* context) : OpKernel(context) {
    
  }
  
  void Compute(OpKernelContext* context) override {
    
    
    const Tensor& grad_out = context->input(0);
    const Tensor& out = context->input(1);
    const Tensor& ipt = context->input(2);
    const Tensor& ii = context->input(3);
    
    
    const TensorShape& grad_out_shape = grad_out.shape();
    const TensorShape& out_shape = out.shape();
    const TensorShape& ipt_shape = ipt.shape();
    const TensorShape& ii_shape = ii.shape();
    
    
    DCHECK_EQ(grad_out_shape.dims(), 1);
    DCHECK_EQ(out_shape.dims(), 1);
    DCHECK_EQ(ipt_shape.dims(), 1);
    DCHECK_EQ(ii_shape.dims(), 1);

    // extra check
    // int m = Example.dim_size(0);
        
    // create output shape
    
    TensorShape grad_ipt_shape(ipt_shape);
    TensorShape grad_ii_shape(ii_shape);
            
    // create output tensor
    
    Tensor* grad_ipt = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_ipt_shape, &grad_ipt));
    Tensor* grad_ii = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, grad_ii_shape, &grad_ii));
    
    // get the corresponding Eigen tensors for data access
    
    auto ipt_tensor = ipt.flat<double>().data();
    auto ii_tensor = ii.flat<int64>().data();
    auto grad_out_tensor = grad_out.flat<double>().data();
    auto out_tensor = out.flat<double>().data();
    auto grad_ipt_tensor = grad_ipt->flat<double>().data();   

    // implement your backward function here 

    // TODO:
    
  }
};
REGISTER_KERNEL_BUILDER(Name("GatherOpsGrad").Device(DEVICE_GPU), GatherOpsGradOpGPU);

#endif