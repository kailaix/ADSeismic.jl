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
void forward(double *out, const double *in, const int64 *srci, const int64* srcj, const double *srcv, int nsrc, int64 NX, int64 NY);
void backward(double *grad_src, double *grad_in, const double *grad_out, const double *in, const int64 *srci, const int64* srcj,
                 const double *srcv, int nsrc, int64 NX, int64 NY);
void forwardGPU(double *out, const double *in, const int64 *srci, const int64* srcj, 
              const double *srcv, int nsrc, const int64 *nx_tensor, const int64 *ny_tensor);
void backwardGPU(double *grad_src, double *grad_in, const double *grad_out, const double *in, const int64 *srci,
   const int64* srcj, const double *srcv, int nsrc, const int64 *nx_tensor, const int64 *ny_tensor);

#if 1
REGISTER_OP("AcousticSource")
.Input("u : double")
  .Input("srci : int64")
  .Input("srcj : int64")
  .Input("srcv : double")
  .Input("nx : int64")
  .Input("ny : int64")
  .Output("out : double")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    
        shape_inference::ShapeHandle u_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &u_shape));
        shape_inference::ShapeHandle srci_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &srci_shape));
        shape_inference::ShapeHandle srcj_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &srcj_shape));
        shape_inference::ShapeHandle srcv_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 1, &srcv_shape));
        shape_inference::ShapeHandle nx_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(4), 0, &nx_shape));
        shape_inference::ShapeHandle ny_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(5), 0, &ny_shape));

        c->set_output(0, c->Vector(-1));
    return Status::OK();
  });


REGISTER_OP("AcousticSourceGrad")
  
  .Input("grad_out : double")
  .Input("out : double")
  .Input("u : double")
  .Input("srci : int64")
  .Input("srcj : int64")
  .Input("srcv : double")
  .Input("nx : int64")
  .Input("ny : int64")
  .Output("grad_u : double")
  .Output("grad_srci : int64")
  .Output("grad_srcj : int64")
  .Output("grad_srcv : double")
  .Output("grad_nx : int64")
  .Output("grad_ny : int64");

class AcousticSourceOp : public OpKernel {
private:
  
public:
  explicit AcousticSourceOp(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {    
    DCHECK_EQ(6, context->num_inputs());
    
    
    const Tensor& u = context->input(0);
    const Tensor& srci = context->input(1);
    const Tensor& srcj = context->input(2);
    const Tensor& srcv = context->input(3);
    const Tensor& nx = context->input(4);
    const Tensor& ny = context->input(5);
    
    
    const TensorShape& u_shape = u.shape();
    const TensorShape& srci_shape = srci.shape();
    const TensorShape& srcj_shape = srcj.shape();
    const TensorShape& srcv_shape = srcv.shape();
    const TensorShape& nx_shape = nx.shape();
    const TensorShape& ny_shape = ny.shape();
    
    
    DCHECK_EQ(u_shape.dims(), 1);
    DCHECK_EQ(srci_shape.dims(), 1);
    DCHECK_EQ(srcj_shape.dims(), 1);
    DCHECK_EQ(srcv_shape.dims(), 1);
    DCHECK_EQ(nx_shape.dims(), 0);
    DCHECK_EQ(ny_shape.dims(), 0);

    // extra check
        
    // create output shape
    

    // extra check
        
    // create output shape
    int nsrc = srcv_shape.dim_size(0);
    TensorShape out_shape({u_shape.dim_size(0)});
            
    // create output tensor
    
    Tensor* out = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &out));
    
    // get the corresponding Eigen tensors for data access
    
    auto u_tensor = u.flat<double>().data();
    auto srci_tensor = srci.flat<int64>().data();
    auto srcj_tensor = srcj.flat<int64>().data();
    auto srcv_tensor = srcv.flat<double>().data();
    auto nx_tensor = nx.flat<int64>().data();
    auto ny_tensor = ny.flat<int64>().data();
    auto out_tensor = out->flat<double>().data();   

    // implement your forward function here 

    // TODO:
    DCHECK_EQ(u_shape.dim_size(0), (*nx_tensor+2)*(*ny_tensor+2));
    forward(out_tensor, u_tensor, srci_tensor, srcj_tensor, srcv_tensor, nsrc, *nx_tensor, *ny_tensor);
  }
};
REGISTER_KERNEL_BUILDER(Name("AcousticSource").Device(DEVICE_CPU), AcousticSourceOp);



class AcousticSourceGradOp : public OpKernel {
private:
  
public:
  explicit AcousticSourceGradOp(OpKernelConstruction* context) : OpKernel(context) {
    
  }
  
  void Compute(OpKernelContext* context) override {
    
    
    const Tensor& grad_out = context->input(0);
    const Tensor& out = context->input(1);
    const Tensor& u = context->input(2);
    const Tensor& srci = context->input(3);
    const Tensor& srcj = context->input(4);
    const Tensor& srcv = context->input(5);
    const Tensor& nx = context->input(6);
    const Tensor& ny = context->input(7);
    
    
    const TensorShape& grad_out_shape = grad_out.shape();
    const TensorShape& out_shape = out.shape();
    const TensorShape& u_shape = u.shape();
    const TensorShape& srci_shape = srci.shape();
    const TensorShape& srcj_shape = srcj.shape();
    const TensorShape& srcv_shape = srcv.shape();
    const TensorShape& nx_shape = nx.shape();
    const TensorShape& ny_shape = ny.shape();
    
    
    DCHECK_EQ(grad_out_shape.dims(), 1);
    DCHECK_EQ(out_shape.dims(), 1);
    DCHECK_EQ(u_shape.dims(), 1);
    DCHECK_EQ(srci_shape.dims(), 1);
    DCHECK_EQ(srcj_shape.dims(), 1);
    DCHECK_EQ(srcv_shape.dims(), 1);
    DCHECK_EQ(nx_shape.dims(), 0);
    DCHECK_EQ(ny_shape.dims(), 0);

    // extra check
    // int m = Example.dim_size(0);
        
    // create output shape
    
    TensorShape grad_u_shape(u_shape);
    TensorShape grad_srci_shape(srci_shape);
    TensorShape grad_srcj_shape(srcj_shape);
    TensorShape grad_srcv_shape(srcv_shape);
    TensorShape grad_nx_shape(nx_shape);
    TensorShape grad_ny_shape(ny_shape);
            
    // create output tensor
    
    Tensor* grad_u = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_u_shape, &grad_u));
    Tensor* grad_srci = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, grad_srci_shape, &grad_srci));
    Tensor* grad_srcj = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(2, grad_srcj_shape, &grad_srcj));
    Tensor* grad_srcv = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(3, grad_srcv_shape, &grad_srcv));
    Tensor* grad_nx = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(4, grad_nx_shape, &grad_nx));
    Tensor* grad_ny = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(5, grad_ny_shape, &grad_ny));
    
    // get the corresponding Eigen tensors for data access
    
    auto u_tensor = u.flat<double>().data();
    auto srci_tensor = srci.flat<int64>().data();
    auto srcj_tensor = srcj.flat<int64>().data();
    auto srcv_tensor = srcv.flat<double>().data();
    auto nx_tensor = nx.flat<int64>().data();
    auto ny_tensor = ny.flat<int64>().data();
    auto grad_out_tensor = grad_out.flat<double>().data();
    auto out_tensor = out.flat<double>().data();
    auto grad_u_tensor = grad_u->flat<double>().data();
    auto grad_srci_tensor = grad_srci->flat<int64>().data();
    auto grad_srcj_tensor = grad_srcj->flat<int64>().data();
    auto grad_srcv_tensor = grad_srcv->flat<double>().data();
    auto grad_nx_tensor = grad_nx->flat<int64>().data();
    auto grad_ny_tensor = grad_ny->flat<int64>().data();   

    // implement your backward function here 

    // TODO:
    int nsrc = srcv_shape.dim_size(0);
    backward(grad_srcv_tensor, grad_u_tensor, grad_out_tensor, u_tensor, srci_tensor, srcj_tensor, srcv_tensor, nsrc, *nx_tensor, *ny_tensor);
  }
};
REGISTER_KERNEL_BUILDER(Name("AcousticSourceGrad").Device(DEVICE_CPU), AcousticSourceGradOp);
#endif


#ifndef NOGPU

class AcousticSourceOpGPU : public OpKernel {
private:
  
public:
  explicit AcousticSourceOpGPU(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {    
    DCHECK_EQ(6, context->num_inputs());
    
    
    const Tensor& u = context->input(0);
    const Tensor& srci = context->input(1);
    const Tensor& srcj = context->input(2);
    const Tensor& srcv = context->input(3);
    const Tensor& nx = context->input(4);
    const Tensor& ny = context->input(5);
    
    
    const TensorShape& u_shape = u.shape();
    const TensorShape& srci_shape = srci.shape();
    const TensorShape& srcj_shape = srcj.shape();
    const TensorShape& srcv_shape = srcv.shape();
    const TensorShape& nx_shape = nx.shape();
    const TensorShape& ny_shape = ny.shape();
    
    
    DCHECK_EQ(u_shape.dims(), 1);
    DCHECK_EQ(srci_shape.dims(), 1);
    DCHECK_EQ(srcj_shape.dims(), 1);
    DCHECK_EQ(srcv_shape.dims(), 1);
    DCHECK_EQ(nx_shape.dims(), 0);
    DCHECK_EQ(ny_shape.dims(), 0);

    // extra check
        
    // create output shape
    

    // extra check
        
    // create output shape
    int nsrc = srcv_shape.dim_size(0);
    TensorShape out_shape({u_shape.dim_size(0)});
            
    // create output tensor
    
    Tensor* out = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &out));
    
    // get the corresponding Eigen tensors for data access
    
    auto u_tensor = u.flat<double>().data();
    auto srci_tensor = srci.flat<int64>().data();
    auto srcj_tensor = srcj.flat<int64>().data();
    auto srcv_tensor = srcv.flat<double>().data();
    auto nx_tensor = nx.flat<int64>().data();
    auto ny_tensor = ny.flat<int64>().data();
    auto out_tensor = out->flat<double>().data();   

    // implement your forward function here 

    // TODO:
    // DCHECK_EQ(u_shape.dim_size(0), (*nx_tensor+2)*(*ny_tensor+2));

    forwardGPU(out_tensor, u_tensor, srci_tensor, srcj_tensor, srcv_tensor, nsrc, nx_tensor, ny_tensor);
  }
};
REGISTER_KERNEL_BUILDER(Name("AcousticSource").Device(DEVICE_GPU), AcousticSourceOpGPU);



class AcousticSourceGradOpGPU : public OpKernel {
private:
  
public:
  explicit AcousticSourceGradOpGPU(OpKernelConstruction* context) : OpKernel(context) {
    
  }
  
  void Compute(OpKernelContext* context) override {
    
    
    const Tensor& grad_out = context->input(0);
    const Tensor& out = context->input(1);
    const Tensor& u = context->input(2);
    const Tensor& srci = context->input(3);
    const Tensor& srcj = context->input(4);
    const Tensor& srcv = context->input(5);
    const Tensor& nx = context->input(6);
    const Tensor& ny = context->input(7);
    
    
    const TensorShape& grad_out_shape = grad_out.shape();
    const TensorShape& out_shape = out.shape();
    const TensorShape& u_shape = u.shape();
    const TensorShape& srci_shape = srci.shape();
    const TensorShape& srcj_shape = srcj.shape();
    const TensorShape& srcv_shape = srcv.shape();
    const TensorShape& nx_shape = nx.shape();
    const TensorShape& ny_shape = ny.shape();
    
    
    DCHECK_EQ(grad_out_shape.dims(), 1);
    DCHECK_EQ(out_shape.dims(), 1);
    DCHECK_EQ(u_shape.dims(), 1);
    DCHECK_EQ(srci_shape.dims(), 1);
    DCHECK_EQ(srcj_shape.dims(), 1);
    DCHECK_EQ(srcv_shape.dims(), 1);
    DCHECK_EQ(nx_shape.dims(), 0);
    DCHECK_EQ(ny_shape.dims(), 0);

    // extra check
    // int m = Example.dim_size(0);
        
    // create output shape
    
    TensorShape grad_u_shape(u_shape);
    TensorShape grad_srci_shape(srci_shape);
    TensorShape grad_srcj_shape(srcj_shape);
    TensorShape grad_srcv_shape(srcv_shape);
    TensorShape grad_nx_shape(nx_shape);
    TensorShape grad_ny_shape(ny_shape);
            
    // create output tensor
    
    Tensor* grad_u = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_u_shape, &grad_u));
    Tensor* grad_srci = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, grad_srci_shape, &grad_srci));
    Tensor* grad_srcj = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(2, grad_srcj_shape, &grad_srcj));
    Tensor* grad_srcv = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(3, grad_srcv_shape, &grad_srcv));
    Tensor* grad_nx = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(4, grad_nx_shape, &grad_nx));
    Tensor* grad_ny = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(5, grad_ny_shape, &grad_ny));
    
    // get the corresponding Eigen tensors for data access
    
    auto u_tensor = u.flat<double>().data();
    auto srci_tensor = srci.flat<int64>().data();
    auto srcj_tensor = srcj.flat<int64>().data();
    auto srcv_tensor = srcv.flat<double>().data();
    auto nx_tensor = nx.flat<int64>().data();
    auto ny_tensor = ny.flat<int64>().data();
    auto grad_out_tensor = grad_out.flat<double>().data();
    auto out_tensor = out.flat<double>().data();
    auto grad_u_tensor = grad_u->flat<double>().data();
    auto grad_srci_tensor = grad_srci->flat<int64>().data();
    auto grad_srcj_tensor = grad_srcj->flat<int64>().data();
    auto grad_srcv_tensor = grad_srcv->flat<double>().data();
    auto grad_nx_tensor = grad_nx->flat<int64>().data();
    auto grad_ny_tensor = grad_ny->flat<int64>().data();   

    // implement your backward function here 

    int nsrc = srcv_shape.dim_size(0);
    backwardGPU(grad_srcv_tensor, grad_u_tensor, grad_out_tensor, u_tensor, srci_tensor, srcj_tensor, srcv_tensor, nsrc, nx_tensor, ny_tensor);
  }
};
REGISTER_KERNEL_BUILDER(Name("AcousticSourceGrad").Device(DEVICE_GPU), AcousticSourceGradOpGPU);

#endif