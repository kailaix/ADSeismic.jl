#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"
#include<cmath>

// Signatures for GPU kernels here 


using namespace tensorflow;
#include "ScatterNdOps.h"


REGISTER_OP("ScatterNdOps")
.Input("ii : int64")
.Input("vv : double")
.Input("m : int64")
.Output("out : double")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    
        shape_inference::ShapeHandle ii_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &ii_shape));
        shape_inference::ShapeHandle vv_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &vv_shape));
        shape_inference::ShapeHandle m_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &m_shape));

        c->set_output(0, c->Vector(-1));
    return Status::OK();
  });

REGISTER_OP("ScatterNdOpsGrad")
.Input("grad_out : double")
.Input("out : double")
.Input("ii : int64")
.Input("vv : double")
.Input("m : int64")
.Output("grad_ii : int64")
.Output("grad_vv : double")
.Output("grad_m : int64");

/*-------------------------------------------------------------------------------------*/

class ScatterNdOpsOp : public OpKernel {
private:
  
public:
  explicit ScatterNdOpsOp(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {    
    DCHECK_EQ(3, context->num_inputs());
    
    
    const Tensor& ii = context->input(0);
    const Tensor& vv = context->input(1);
    const Tensor& m = context->input(2);
    
    
    const TensorShape& ii_shape = ii.shape();
    const TensorShape& vv_shape = vv.shape();
    const TensorShape& m_shape = m.shape();
    
    
    DCHECK_EQ(ii_shape.dims(), 1);
    DCHECK_EQ(vv_shape.dims(), 1);
    DCHECK_EQ(m_shape.dims(), 0);

    // extra check
        
    // create output shape
    int n = *m.flat<int64>().data();
    int N = vv_shape.dim_size(0);
    TensorShape out_shape({n});
            
    // create output tensor
    
    Tensor* out = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &out));
    
    // get the corresponding Eigen tensors for data access
    
    auto ii_tensor = ii.flat<int64>().data();
    auto vv_tensor = vv.flat<double>().data();
    auto m_tensor = m.flat<int64>().data();
    auto out_tensor = out->flat<double>().data();   

    // implement your forward function here 

    // TODO:
    out->flat<double>().setZero();
    ScatterNdOps_forward(out_tensor, ii_tensor, vv_tensor, N);

  }
};
REGISTER_KERNEL_BUILDER(Name("ScatterNdOps").Device(DEVICE_CPU), ScatterNdOpsOp);



class ScatterNdOpsGradOp : public OpKernel {
private:
  
public:
  explicit ScatterNdOpsGradOp(OpKernelConstruction* context) : OpKernel(context) {
    
  }
  
  void Compute(OpKernelContext* context) override {
    
    
    const Tensor& grad_out = context->input(0);
    const Tensor& out = context->input(1);
    const Tensor& ii = context->input(2);
    const Tensor& vv = context->input(3);
    const Tensor& m = context->input(4);
    
    
    const TensorShape& grad_out_shape = grad_out.shape();
    const TensorShape& out_shape = out.shape();
    const TensorShape& ii_shape = ii.shape();
    const TensorShape& vv_shape = vv.shape();
    const TensorShape& m_shape = m.shape();
    
    
    DCHECK_EQ(grad_out_shape.dims(), 1);
    DCHECK_EQ(out_shape.dims(), 1);
    DCHECK_EQ(ii_shape.dims(), 1);
    DCHECK_EQ(vv_shape.dims(), 1);
    DCHECK_EQ(m_shape.dims(), 0);

    // extra check
    // int m = Example.dim_size(0);
        
    // create output shape
    
    TensorShape grad_ii_shape(ii_shape);
    TensorShape grad_vv_shape(vv_shape);
    TensorShape grad_m_shape(m_shape);
            
    // create output tensor
    
    Tensor* grad_ii = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_ii_shape, &grad_ii));
    Tensor* grad_vv = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, grad_vv_shape, &grad_vv));
    Tensor* grad_m = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(2, grad_m_shape, &grad_m));
    
    // get the corresponding Eigen tensors for data access
    
    auto ii_tensor = ii.flat<int64>().data();
    auto vv_tensor = vv.flat<double>().data();
    auto m_tensor = m.flat<int64>().data();
    auto grad_out_tensor = grad_out.flat<double>().data();
    auto out_tensor = out.flat<double>().data();
    auto grad_vv_tensor = grad_vv->flat<double>().data();   

    // implement your backward function here 

    // TODO:
    int N = vv_shape.dim_size(0);
    grad_vv->flat<double>().setZero();
    ScatterNdOps_backward(
      grad_vv_tensor, grad_out_tensor, out_tensor, ii_tensor, vv_tensor, N);
    
  }
};
REGISTER_KERNEL_BUILDER(Name("ScatterNdOpsGrad").Device(DEVICE_CPU), ScatterNdOpsGradOp);


/***************************************************************************************
**********************            GPU Operator            ******************************
***************************************************************************************/


#ifdef GOOGLE_CUDA
class ScatterNdOpsOpGPU : public OpKernel {
private:
  
public:
  explicit ScatterNdOpsOpGPU(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {    
    DCHECK_EQ(3, context->num_inputs());
    
    
    const Tensor& ii = context->input(0);
    const Tensor& vv = context->input(1);
    const Tensor& m = context->input(2);
    
    
    const TensorShape& ii_shape = ii.shape();
    const TensorShape& vv_shape = vv.shape();
    const TensorShape& m_shape = m.shape();
    
    
    DCHECK_EQ(ii_shape.dims(), 1);
    DCHECK_EQ(vv_shape.dims(), 1);
    DCHECK_EQ(m_shape.dims(), 0);

    // extra check
        
    // create output shape
    int64 n64;
    get_ScatterNdOps_num(&n64, m.flat<int64>().data());
    int N = vv_shape.dim_size(0);
    TensorShape out_shape({n64});


    // create output tensor
    Tensor* out = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &out));
    
    // get the corresponding Eigen tensors for data access
    
    auto ii_tensor = ii.flat<int64>().data();
    auto vv_tensor = vv.flat<double>().data();
    auto m_tensor = m.flat<int64>().data();
    auto out_tensor = out->flat<double>().data();   

    // implement your forward function here 

    // TODO:
    out->flat<double>().setZero();
    ScatterNdOps_forward(out_tensor, ii_tensor, vv_tensor, N);

  }
};
REGISTER_KERNEL_BUILDER(Name("ScatterNdOps").Device(DEVICE_GPU), ScatterNdOpsOpGPU);

class ScatterNdOpsGradOpGPU : public OpKernel {
private:
  
public:
  explicit ScatterNdOpsGradOpGPU(OpKernelConstruction* context) : OpKernel(context) {
    
  }
  
  void Compute(OpKernelContext* context) override {
    
    
    const Tensor& grad_out = context->input(0);
    const Tensor& out = context->input(1);
    const Tensor& ii = context->input(2);
    const Tensor& vv = context->input(3);
    const Tensor& m = context->input(4);
    
    
    const TensorShape& grad_out_shape = grad_out.shape();
    const TensorShape& out_shape = out.shape();
    const TensorShape& ii_shape = ii.shape();
    const TensorShape& vv_shape = vv.shape();
    const TensorShape& m_shape = m.shape();
    
    
    DCHECK_EQ(grad_out_shape.dims(), 1);
    DCHECK_EQ(out_shape.dims(), 1);
    DCHECK_EQ(ii_shape.dims(), 1);
    DCHECK_EQ(vv_shape.dims(), 1);
    DCHECK_EQ(m_shape.dims(), 0);

    // extra check
    // int m = Example.dim_size(0);
        
    // create output shape
    
    TensorShape grad_ii_shape(ii_shape);
    TensorShape grad_vv_shape(vv_shape);
    TensorShape grad_m_shape(m_shape);
            
    // create output tensor
    
    Tensor* grad_ii = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_ii_shape, &grad_ii));
    Tensor* grad_vv = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, grad_vv_shape, &grad_vv));
    Tensor* grad_m = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(2, grad_m_shape, &grad_m));
    
    // get the corresponding Eigen tensors for data access
    
    auto ii_tensor = ii.flat<int64>().data();
    auto vv_tensor = vv.flat<double>().data();
    auto m_tensor = m.flat<int64>().data();
    auto grad_out_tensor = grad_out.flat<double>().data();
    auto out_tensor = out.flat<double>().data();
    auto grad_vv_tensor = grad_vv->flat<double>().data();   

    // implement your backward function here 

    // TODO:
    int N = vv_shape.dim_size(0);
    grad_vv->flat<double>().setZero();
    ScatterNdOps_backward(
      grad_vv_tensor, grad_out_tensor, out_tensor, ii_tensor, vv_tensor, N);
    
  }
};
REGISTER_KERNEL_BUILDER(Name("ScatterNdOpsGrad").Device(DEVICE_GPU), ScatterNdOpsGradOpGPU);

#endif