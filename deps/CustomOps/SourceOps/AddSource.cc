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
// #include "AddSource.h"
void forwardGPU(
    double * vx_,
    double * vy_,
    double * sigmaxx_,
    double * sigmayy_,
    double * sigmaxy_,
    const double * vx,
    const double * vy,
    const double * sigmaxx,
    const double * sigmayy,
    const double * sigmaxy,
    const long long  * srci, const long long  *srcj, const double *srcv, const long long  *src_type, long long  nsrc,
    const long long *  NX, const long long *NY
);

void forwardCPU(
    double * vx_,
    double * vy_,
    double * sigmaxx_,
    double * sigmayy_,
    double * sigmaxy_,
    const double * vx,
    const double * vy,
    const double * sigmaxx,
    const double * sigmayy,
    const double * sigmaxy,
    const int64 * srci, const int64 *srcj, const double *srcv, const int64 *src_type, int64 nsrc,
    int64 NX, int64 NY
);

void backwardCPU(
    double * g_vx,
    double * g_vy,
    double * g_sigmaxx,
    double * g_sigmayy,
    double * g_sigmaxy,
    double * grad_srcv,
    const double * g_vx_,
    const double * g_vy_,
    const double * g_sigmaxx_,
    const double * g_sigmayy_,
    const double * g_sigmaxy_,
    const int64 * srci, const int64 *srcj, const double *srcv, const int64 *src_type, int64 nsrc,
    int64 NX, int64 NY
);

void backwardGPU(
    double * g_vx,
    double * g_vy,
    double * g_sigmaxx,
    double * g_sigmayy,
    double * g_sigmaxy,
    double * grad_srcv,
    const double * g_vx_,
    const double * g_vy_,
    const double * g_sigmaxx_,
    const double * g_sigmayy_,
    const double * g_sigmaxy_,
    const long long  * srci, const long long  *srcj, const double *srcv, const long long  *src_type, long long  nsrc,
    const long long *    nx, const long long *    ny
);

REGISTER_OP("AddSource")
.Input("sigmaxx : double")
  .Input("sigmayy : double")
  .Input("sigmaxy : double")
  .Input("vx : double")
  .Input("vy : double")
  .Input("srci : int64")
  .Input("srcj : int64")
  .Input("srctype : int64")
  .Input("nx : int64")
  .Input("ny : int64")
  .Input("srcv : double")
  .Output("sigmaxxout : double")
  .Output("sigmayyout : double")
  .Output("sigmaxyout : double")
  .Output("vxout : double")
  .Output("vyout : double")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    
        shape_inference::ShapeHandle sigmaxx_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &sigmaxx_shape));
        shape_inference::ShapeHandle sigmayy_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &sigmayy_shape));
        shape_inference::ShapeHandle sigmaxy_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &sigmaxy_shape));
        shape_inference::ShapeHandle vx_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 1, &vx_shape));
        shape_inference::ShapeHandle vy_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(4), 1, &vy_shape));
        shape_inference::ShapeHandle srci_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(5), 1, &srci_shape));
        shape_inference::ShapeHandle srcj_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(6), 1, &srcj_shape));
        shape_inference::ShapeHandle srctype_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(7), 1, &srctype_shape));
        shape_inference::ShapeHandle nx_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(8), 0, &nx_shape));
        shape_inference::ShapeHandle ny_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(9), 0, &ny_shape));
        shape_inference::ShapeHandle srcv_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(10), 1, &srcv_shape));

        c->set_output(0, c->input(0));
        c->set_output(1, c->input(0));
        c->set_output(2, c->input(0));
        c->set_output(3, c->input(0));
        c->set_output(4, c->input(0));
    return Status::OK();
  });

REGISTER_OP("AddSourceGrad")
  
  .Input("grad_sigmaxxout : double")
.Input("grad_sigmayyout : double")
.Input("grad_sigmaxyout : double")
.Input("grad_vxout : double")
.Input("grad_vyout : double")
  .Input("sigmaxxout : double")
  .Input("sigmayyout : double")
  .Input("sigmaxyout : double")
  .Input("vxout : double")
  .Input("vyout : double")
  .Input("sigmaxx : double")
  .Input("sigmayy : double")
  .Input("sigmaxy : double")
  .Input("vx : double")
  .Input("vy : double")
  .Input("srci : int64")
  .Input("srcj : int64")
  .Input("srctype : int64")
  .Input("nx : int64")
  .Input("ny : int64")
  .Input("srcv : double")
  .Output("grad_sigmaxx : double")
  .Output("grad_sigmayy : double")
  .Output("grad_sigmaxy : double")
  .Output("grad_vx : double")
  .Output("grad_vy : double")
  .Output("grad_srci : int64")
  .Output("grad_srcj : int64")
  .Output("grad_srctype : int64")
  .Output("grad_nx : int64")
  .Output("grad_ny : int64")
  .Output("grad_srcv : double");


#ifndef NOGPU


class AddSourceOpGPU : public OpKernel {
private:
  
public:
  explicit AddSourceOpGPU(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {    
    DCHECK_EQ(11, context->num_inputs());
    
    
    const Tensor& sigmaxx = context->input(0);
    const Tensor& sigmayy = context->input(1);
    const Tensor& sigmaxy = context->input(2);
    const Tensor& vx = context->input(3);
    const Tensor& vy = context->input(4);
    const Tensor& srci = context->input(5);
    const Tensor& srcj = context->input(6);
    const Tensor& srctype = context->input(7);
    const Tensor& nx = context->input(8);
    const Tensor& ny = context->input(9);
    const Tensor& srcv = context->input(10);
    
    
    const TensorShape& sigmaxx_shape = sigmaxx.shape();
    const TensorShape& sigmayy_shape = sigmayy.shape();
    const TensorShape& sigmaxy_shape = sigmaxy.shape();
    const TensorShape& vx_shape = vx.shape();
    const TensorShape& vy_shape = vy.shape();
    const TensorShape& srci_shape = srci.shape();
    const TensorShape& srcj_shape = srcj.shape();
    const TensorShape& srctype_shape = srctype.shape();
    const TensorShape& nx_shape = nx.shape();
    const TensorShape& ny_shape = ny.shape();
    const TensorShape& srcv_shape = srcv.shape();
    
    
    DCHECK_EQ(sigmaxx_shape.dims(), 1);
    DCHECK_EQ(sigmayy_shape.dims(), 1);
    DCHECK_EQ(sigmaxy_shape.dims(), 1);
    DCHECK_EQ(vx_shape.dims(), 1);
    DCHECK_EQ(vy_shape.dims(), 1);
    DCHECK_EQ(srci_shape.dims(), 1);
    DCHECK_EQ(srcj_shape.dims(), 1);
    DCHECK_EQ(srctype_shape.dims(), 1);
    DCHECK_EQ(nx_shape.dims(), 0);
    DCHECK_EQ(ny_shape.dims(), 0);
    DCHECK_EQ(srcv_shape.dims(), 1);

    // extra check
        
    // create output shape
    int nsrc = srcv_shape.dim_size(0);
    int N = sigmaxx_shape.dim_size(0);
    TensorShape sigmaxxout_shape({N});
    TensorShape sigmayyout_shape({N});
    TensorShape sigmaxyout_shape({N});
    TensorShape vxout_shape({N});
    TensorShape vyout_shape({N});
            
    // create output tensor
    
    Tensor* sigmaxxout = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, sigmaxxout_shape, &sigmaxxout));
    Tensor* sigmayyout = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, sigmayyout_shape, &sigmayyout));
    Tensor* sigmaxyout = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(2, sigmaxyout_shape, &sigmaxyout));
    Tensor* vxout = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(3, vxout_shape, &vxout));
    Tensor* vyout = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(4, vyout_shape, &vyout));
    
    // get the corresponding Eigen tensors for data access
    
    auto sigmaxx_tensor = sigmaxx.flat<double>().data();
    auto sigmayy_tensor = sigmayy.flat<double>().data();
    auto sigmaxy_tensor = sigmaxy.flat<double>().data();
    auto vx_tensor = vx.flat<double>().data();
    auto vy_tensor = vy.flat<double>().data();
    auto srci_tensor = srci.flat<int64>().data();
    auto srcj_tensor = srcj.flat<int64>().data();
    auto srctype_tensor = srctype.flat<int64>().data();
    auto nx_tensor = nx.flat<int64>().data();
    auto ny_tensor = ny.flat<int64>().data();
    auto srcv_tensor = srcv.flat<double>().data();
    auto sigmaxxout_tensor = sigmaxxout->flat<double>().data();
    auto sigmayyout_tensor = sigmayyout->flat<double>().data();
    auto sigmaxyout_tensor = sigmaxyout->flat<double>().data();
    auto vxout_tensor = vxout->flat<double>().data();
    auto vyout_tensor = vyout->flat<double>().data();   

    // implement your forward function here 
    // std::cout << vx.flat<double>() << std::endl;
    forwardGPU(
      vxout_tensor,
      vyout_tensor,
      sigmaxxout_tensor,
      sigmayyout_tensor,
      sigmaxyout_tensor,
      vx_tensor,
      vy_tensor,
      sigmaxx_tensor,
      sigmayy_tensor,
      sigmaxy_tensor,
      srci_tensor, 
      srcj_tensor,
      srcv_tensor, 
      srctype_tensor,
      nsrc, nx_tensor, ny_tensor
    );

  }
};
REGISTER_KERNEL_BUILDER(Name("AddSource").Device(DEVICE_GPU), AddSourceOpGPU);





class AddSourceGradOpGPU : public OpKernel {
private:
  
public:
  explicit AddSourceGradOpGPU(OpKernelConstruction* context) : OpKernel(context) {
    
  }
  
  void Compute(OpKernelContext* context) override {
    
    
    const Tensor& grad_sigmaxxout = context->input(0);
    const Tensor& grad_sigmayyout = context->input(1);
    const Tensor& grad_sigmaxyout = context->input(2);
    const Tensor& grad_vxout = context->input(3);
    const Tensor& grad_vyout = context->input(4);
    const Tensor& sigmaxxout = context->input(5);
    const Tensor& sigmayyout = context->input(6);
    const Tensor& sigmaxyout = context->input(7);
    const Tensor& vxout = context->input(8);
    const Tensor& vyout = context->input(9);
    const Tensor& sigmaxx = context->input(10);
    const Tensor& sigmayy = context->input(11);
    const Tensor& sigmaxy = context->input(12);
    const Tensor& vx = context->input(13);
    const Tensor& vy = context->input(14);
    const Tensor& srci = context->input(15);
    const Tensor& srcj = context->input(16);
    const Tensor& srctype = context->input(17);
    const Tensor& nx = context->input(18);
    const Tensor& ny = context->input(19);
    const Tensor& srcv = context->input(20);
    
    
    const TensorShape& grad_sigmaxxout_shape = grad_sigmaxxout.shape();
    const TensorShape& grad_sigmayyout_shape = grad_sigmayyout.shape();
    const TensorShape& grad_sigmaxyout_shape = grad_sigmaxyout.shape();
    const TensorShape& grad_vxout_shape = grad_vxout.shape();
    const TensorShape& grad_vyout_shape = grad_vyout.shape();
    const TensorShape& sigmaxxout_shape = sigmaxxout.shape();
    const TensorShape& sigmayyout_shape = sigmayyout.shape();
    const TensorShape& sigmaxyout_shape = sigmaxyout.shape();
    const TensorShape& vxout_shape = vxout.shape();
    const TensorShape& vyout_shape = vyout.shape();
    const TensorShape& sigmaxx_shape = sigmaxx.shape();
    const TensorShape& sigmayy_shape = sigmayy.shape();
    const TensorShape& sigmaxy_shape = sigmaxy.shape();
    const TensorShape& vx_shape = vx.shape();
    const TensorShape& vy_shape = vy.shape();
    const TensorShape& srci_shape = srci.shape();
    const TensorShape& srcj_shape = srcj.shape();
    const TensorShape& srctype_shape = srctype.shape();
    const TensorShape& nx_shape = nx.shape();
    const TensorShape& ny_shape = ny.shape();
    const TensorShape& srcv_shape = srcv.shape();
    
    
    DCHECK_EQ(grad_sigmaxxout_shape.dims(), 1);
    DCHECK_EQ(grad_sigmayyout_shape.dims(), 1);
    DCHECK_EQ(grad_sigmaxyout_shape.dims(), 1);
    DCHECK_EQ(grad_vxout_shape.dims(), 1);
    DCHECK_EQ(grad_vyout_shape.dims(), 1);
    DCHECK_EQ(sigmaxxout_shape.dims(), 1);
    DCHECK_EQ(sigmayyout_shape.dims(), 1);
    DCHECK_EQ(sigmaxyout_shape.dims(), 1);
    DCHECK_EQ(vxout_shape.dims(), 1);
    DCHECK_EQ(vyout_shape.dims(), 1);
    DCHECK_EQ(sigmaxx_shape.dims(), 1);
    DCHECK_EQ(sigmayy_shape.dims(), 1);
    DCHECK_EQ(sigmaxy_shape.dims(), 1);
    DCHECK_EQ(vx_shape.dims(), 1);
    DCHECK_EQ(vy_shape.dims(), 1);
    DCHECK_EQ(srci_shape.dims(), 1);
    DCHECK_EQ(srcj_shape.dims(), 1);
    DCHECK_EQ(srctype_shape.dims(), 1);
    DCHECK_EQ(nx_shape.dims(), 0);
    DCHECK_EQ(ny_shape.dims(), 0);
    DCHECK_EQ(srcv_shape.dims(), 1);

    // extra check
    // int m = Example.dim_size(0);
        
    // create output shape
    
    TensorShape grad_sigmaxx_shape(sigmaxx_shape);
    TensorShape grad_sigmayy_shape(sigmayy_shape);
    TensorShape grad_sigmaxy_shape(sigmaxy_shape);
    TensorShape grad_vx_shape(vx_shape);
    TensorShape grad_vy_shape(vy_shape);
    TensorShape grad_srci_shape(srci_shape);
    TensorShape grad_srcj_shape(srcj_shape);
    TensorShape grad_srctype_shape(srctype_shape);
    TensorShape grad_nx_shape(nx_shape);
    TensorShape grad_ny_shape(ny_shape);
    TensorShape grad_srcv_shape(srcv_shape);
            
    // create output tensor
    
    Tensor* grad_sigmaxx = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_sigmaxx_shape, &grad_sigmaxx));
    Tensor* grad_sigmayy = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, grad_sigmayy_shape, &grad_sigmayy));
    Tensor* grad_sigmaxy = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(2, grad_sigmaxy_shape, &grad_sigmaxy));
    Tensor* grad_vx = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(3, grad_vx_shape, &grad_vx));
    Tensor* grad_vy = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(4, grad_vy_shape, &grad_vy));
    Tensor* grad_srci = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(5, grad_srci_shape, &grad_srci));
    Tensor* grad_srcj = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(6, grad_srcj_shape, &grad_srcj));
    Tensor* grad_srctype = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(7, grad_srctype_shape, &grad_srctype));
    Tensor* grad_nx = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(8, grad_nx_shape, &grad_nx));
    Tensor* grad_ny = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(9, grad_ny_shape, &grad_ny));
    Tensor* grad_srcv = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(10, grad_srcv_shape, &grad_srcv));
    
    // get the corresponding Eigen tensors for data access
    
    auto sigmaxx_tensor = sigmaxx.flat<double>().data();
    auto sigmayy_tensor = sigmayy.flat<double>().data();
    auto sigmaxy_tensor = sigmaxy.flat<double>().data();
    auto vx_tensor = vx.flat<double>().data();
    auto vy_tensor = vy.flat<double>().data();
    auto srci_tensor = srci.flat<int64>().data();
    auto srcj_tensor = srcj.flat<int64>().data();
    auto srctype_tensor = srctype.flat<int64>().data();
    auto nx_tensor = nx.flat<int64>().data();
    auto ny_tensor = ny.flat<int64>().data();
    auto srcv_tensor = srcv.flat<double>().data();
    auto grad_sigmaxxout_tensor = grad_sigmaxxout.flat<double>().data();
    auto grad_sigmayyout_tensor = grad_sigmayyout.flat<double>().data();
    auto grad_sigmaxyout_tensor = grad_sigmaxyout.flat<double>().data();
    auto grad_vxout_tensor = grad_vxout.flat<double>().data();
    auto grad_vyout_tensor = grad_vyout.flat<double>().data();
    auto sigmaxxout_tensor = sigmaxxout.flat<double>().data();
    auto sigmayyout_tensor = sigmayyout.flat<double>().data();
    auto sigmaxyout_tensor = sigmaxyout.flat<double>().data();
    auto vxout_tensor = vxout.flat<double>().data();
    auto vyout_tensor = vyout.flat<double>().data();
    auto grad_sigmaxx_tensor = grad_sigmaxx->flat<double>().data();
    auto grad_sigmayy_tensor = grad_sigmayy->flat<double>().data();
    auto grad_sigmaxy_tensor = grad_sigmaxy->flat<double>().data();
    auto grad_vx_tensor = grad_vx->flat<double>().data();
    auto grad_vy_tensor = grad_vy->flat<double>().data();
    auto grad_srci_tensor = grad_srci->flat<int64>().data();
    auto grad_srcj_tensor = grad_srcj->flat<int64>().data();
    auto grad_srctype_tensor = grad_srctype->flat<int64>().data();
    auto grad_nx_tensor = grad_nx->flat<int64>().data();
    auto grad_ny_tensor = grad_ny->flat<int64>().data();
    auto grad_srcv_tensor = grad_srcv->flat<double>().data();   

    // implement your backward function here 

    // TODO:
    int64 nsrc = srcv_shape.dim_size(0);
    backwardGPU(
      grad_vx_tensor,
      grad_vy_tensor,
      grad_sigmaxx_tensor,
      grad_sigmayy_tensor,
      grad_sigmaxy_tensor,
      grad_srcv_tensor,
      grad_vxout_tensor,
      grad_vyout_tensor,
      grad_sigmaxxout_tensor,
      grad_sigmayyout_tensor,
      grad_sigmaxyout_tensor,
      srci_tensor, 
      srcj_tensor,
      srcv_tensor, 
      srctype_tensor,
      nsrc, nx_tensor, ny_tensor
    );
  

  }
};
REGISTER_KERNEL_BUILDER(Name("AddSourceGrad").Device(DEVICE_GPU), AddSourceGradOpGPU);

#endif

class AddSourceOpCPU : public OpKernel {
private:
  
public:
  explicit AddSourceOpCPU(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {    
    DCHECK_EQ(11, context->num_inputs());
    
    
    const Tensor& sigmaxx = context->input(0);
    const Tensor& sigmayy = context->input(1);
    const Tensor& sigmaxy = context->input(2);
    const Tensor& vx = context->input(3);
    const Tensor& vy = context->input(4);
    const Tensor& srci = context->input(5);
    const Tensor& srcj = context->input(6);
    const Tensor& srctype = context->input(7);
    const Tensor& nx = context->input(8);
    const Tensor& ny = context->input(9);
    const Tensor& srcv = context->input(10);
    
    
    const TensorShape& sigmaxx_shape = sigmaxx.shape();
    const TensorShape& sigmayy_shape = sigmayy.shape();
    const TensorShape& sigmaxy_shape = sigmaxy.shape();
    const TensorShape& vx_shape = vx.shape();
    const TensorShape& vy_shape = vy.shape();
    const TensorShape& srci_shape = srci.shape();
    const TensorShape& srcj_shape = srcj.shape();
    const TensorShape& srctype_shape = srctype.shape();
    const TensorShape& nx_shape = nx.shape();
    const TensorShape& ny_shape = ny.shape();
    const TensorShape& srcv_shape = srcv.shape();
    
    
    DCHECK_EQ(sigmaxx_shape.dims(), 1);
    DCHECK_EQ(sigmayy_shape.dims(), 1);
    DCHECK_EQ(sigmaxy_shape.dims(), 1);
    DCHECK_EQ(vx_shape.dims(), 1);
    DCHECK_EQ(vy_shape.dims(), 1);
    DCHECK_EQ(srci_shape.dims(), 1);
    DCHECK_EQ(srcj_shape.dims(), 1);
    DCHECK_EQ(srctype_shape.dims(), 1);
    DCHECK_EQ(nx_shape.dims(), 0);
    DCHECK_EQ(ny_shape.dims(), 0);
    DCHECK_EQ(srcv_shape.dims(), 1);

    // extra check
        
    // create output shape
    int nsrc = srcv_shape.dim_size(0);
    int N = sigmaxx_shape.dim_size(0);
    TensorShape sigmaxxout_shape({N});
    TensorShape sigmayyout_shape({N});
    TensorShape sigmaxyout_shape({N});
    TensorShape vxout_shape({N});
    TensorShape vyout_shape({N});
            
    // create output tensor
    
    Tensor* sigmaxxout = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, sigmaxxout_shape, &sigmaxxout));
    Tensor* sigmayyout = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, sigmayyout_shape, &sigmayyout));
    Tensor* sigmaxyout = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(2, sigmaxyout_shape, &sigmaxyout));
    Tensor* vxout = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(3, vxout_shape, &vxout));
    Tensor* vyout = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(4, vyout_shape, &vyout));
    
    // get the corresponding Eigen tensors for data access
    
    auto sigmaxx_tensor = sigmaxx.flat<double>().data();
    auto sigmayy_tensor = sigmayy.flat<double>().data();
    auto sigmaxy_tensor = sigmaxy.flat<double>().data();
    auto vx_tensor = vx.flat<double>().data();
    auto vy_tensor = vy.flat<double>().data();
    auto srci_tensor = srci.flat<int64>().data();
    auto srcj_tensor = srcj.flat<int64>().data();
    auto srctype_tensor = srctype.flat<int64>().data();
    auto nx_tensor = nx.flat<int64>().data();
    auto ny_tensor = ny.flat<int64>().data();
    auto srcv_tensor = srcv.flat<double>().data();
    auto sigmaxxout_tensor = sigmaxxout->flat<double>().data();
    auto sigmayyout_tensor = sigmayyout->flat<double>().data();
    auto sigmaxyout_tensor = sigmaxyout->flat<double>().data();
    auto vxout_tensor = vxout->flat<double>().data();
    auto vyout_tensor = vyout->flat<double>().data();   

    // implement your forward function here 

    // TODO:
    int64 NX = *nx.flat<int64>().data();
    int64 NY = *ny.flat<int64>().data();
    forwardCPU(
      vxout_tensor,
      vyout_tensor,
      sigmaxxout_tensor,
      sigmayyout_tensor,
      sigmaxyout_tensor,
      vx_tensor,
      vy_tensor,
      sigmaxx_tensor,
      sigmayy_tensor,
      sigmaxy_tensor,
      srci_tensor, 
      srcj_tensor,
      srcv_tensor, 
      srctype_tensor,
      nsrc, NX, NY
    );

  }
};
REGISTER_KERNEL_BUILDER(Name("AddSource").Device(DEVICE_CPU), AddSourceOpCPU);


class AddSourceGradOp : public OpKernel {
private:
  
public:
  explicit AddSourceGradOp(OpKernelConstruction* context) : OpKernel(context) {
    
  }
  
  void Compute(OpKernelContext* context) override {
    
    
    const Tensor& grad_sigmaxxout = context->input(0);
    const Tensor& grad_sigmayyout = context->input(1);
    const Tensor& grad_sigmaxyout = context->input(2);
    const Tensor& grad_vxout = context->input(3);
    const Tensor& grad_vyout = context->input(4);
    const Tensor& sigmaxxout = context->input(5);
    const Tensor& sigmayyout = context->input(6);
    const Tensor& sigmaxyout = context->input(7);
    const Tensor& vxout = context->input(8);
    const Tensor& vyout = context->input(9);
    const Tensor& sigmaxx = context->input(10);
    const Tensor& sigmayy = context->input(11);
    const Tensor& sigmaxy = context->input(12);
    const Tensor& vx = context->input(13);
    const Tensor& vy = context->input(14);
    const Tensor& srci = context->input(15);
    const Tensor& srcj = context->input(16);
    const Tensor& srctype = context->input(17);
    const Tensor& nx = context->input(18);
    const Tensor& ny = context->input(19);
    const Tensor& srcv = context->input(20);
    
    
    const TensorShape& grad_sigmaxxout_shape = grad_sigmaxxout.shape();
    const TensorShape& grad_sigmayyout_shape = grad_sigmayyout.shape();
    const TensorShape& grad_sigmaxyout_shape = grad_sigmaxyout.shape();
    const TensorShape& grad_vxout_shape = grad_vxout.shape();
    const TensorShape& grad_vyout_shape = grad_vyout.shape();
    const TensorShape& sigmaxxout_shape = sigmaxxout.shape();
    const TensorShape& sigmayyout_shape = sigmayyout.shape();
    const TensorShape& sigmaxyout_shape = sigmaxyout.shape();
    const TensorShape& vxout_shape = vxout.shape();
    const TensorShape& vyout_shape = vyout.shape();
    const TensorShape& sigmaxx_shape = sigmaxx.shape();
    const TensorShape& sigmayy_shape = sigmayy.shape();
    const TensorShape& sigmaxy_shape = sigmaxy.shape();
    const TensorShape& vx_shape = vx.shape();
    const TensorShape& vy_shape = vy.shape();
    const TensorShape& srci_shape = srci.shape();
    const TensorShape& srcj_shape = srcj.shape();
    const TensorShape& srctype_shape = srctype.shape();
    const TensorShape& nx_shape = nx.shape();
    const TensorShape& ny_shape = ny.shape();
    const TensorShape& srcv_shape = srcv.shape();
    
    
    DCHECK_EQ(grad_sigmaxxout_shape.dims(), 1);
    DCHECK_EQ(grad_sigmayyout_shape.dims(), 1);
    DCHECK_EQ(grad_sigmaxyout_shape.dims(), 1);
    DCHECK_EQ(grad_vxout_shape.dims(), 1);
    DCHECK_EQ(grad_vyout_shape.dims(), 1);
    DCHECK_EQ(sigmaxxout_shape.dims(), 1);
    DCHECK_EQ(sigmayyout_shape.dims(), 1);
    DCHECK_EQ(sigmaxyout_shape.dims(), 1);
    DCHECK_EQ(vxout_shape.dims(), 1);
    DCHECK_EQ(vyout_shape.dims(), 1);
    DCHECK_EQ(sigmaxx_shape.dims(), 1);
    DCHECK_EQ(sigmayy_shape.dims(), 1);
    DCHECK_EQ(sigmaxy_shape.dims(), 1);
    DCHECK_EQ(vx_shape.dims(), 1);
    DCHECK_EQ(vy_shape.dims(), 1);
    DCHECK_EQ(srci_shape.dims(), 1);
    DCHECK_EQ(srcj_shape.dims(), 1);
    DCHECK_EQ(srctype_shape.dims(), 1);
    DCHECK_EQ(nx_shape.dims(), 0);
    DCHECK_EQ(ny_shape.dims(), 0);
    DCHECK_EQ(srcv_shape.dims(), 1);

    // extra check
    // int m = Example.dim_size(0);
        
    // create output shape
    
    TensorShape grad_sigmaxx_shape(sigmaxx_shape);
    TensorShape grad_sigmayy_shape(sigmayy_shape);
    TensorShape grad_sigmaxy_shape(sigmaxy_shape);
    TensorShape grad_vx_shape(vx_shape);
    TensorShape grad_vy_shape(vy_shape);
    TensorShape grad_srci_shape(srci_shape);
    TensorShape grad_srcj_shape(srcj_shape);
    TensorShape grad_srctype_shape(srctype_shape);
    TensorShape grad_nx_shape(nx_shape);
    TensorShape grad_ny_shape(ny_shape);
    TensorShape grad_srcv_shape(srcv_shape);
            
    // create output tensor
    
    Tensor* grad_sigmaxx = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_sigmaxx_shape, &grad_sigmaxx));
    Tensor* grad_sigmayy = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, grad_sigmayy_shape, &grad_sigmayy));
    Tensor* grad_sigmaxy = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(2, grad_sigmaxy_shape, &grad_sigmaxy));
    Tensor* grad_vx = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(3, grad_vx_shape, &grad_vx));
    Tensor* grad_vy = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(4, grad_vy_shape, &grad_vy));
    Tensor* grad_srci = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(5, grad_srci_shape, &grad_srci));
    Tensor* grad_srcj = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(6, grad_srcj_shape, &grad_srcj));
    Tensor* grad_srctype = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(7, grad_srctype_shape, &grad_srctype));
    Tensor* grad_nx = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(8, grad_nx_shape, &grad_nx));
    Tensor* grad_ny = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(9, grad_ny_shape, &grad_ny));
    Tensor* grad_srcv = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(10, grad_srcv_shape, &grad_srcv));
    
    // get the corresponding Eigen tensors for data access
    
    auto sigmaxx_tensor = sigmaxx.flat<double>().data();
    auto sigmayy_tensor = sigmayy.flat<double>().data();
    auto sigmaxy_tensor = sigmaxy.flat<double>().data();
    auto vx_tensor = vx.flat<double>().data();
    auto vy_tensor = vy.flat<double>().data();
    auto srci_tensor = srci.flat<int64>().data();
    auto srcj_tensor = srcj.flat<int64>().data();
    auto srctype_tensor = srctype.flat<int64>().data();
    auto nx_tensor = nx.flat<int64>().data();
    auto ny_tensor = ny.flat<int64>().data();
    auto srcv_tensor = srcv.flat<double>().data();
    auto grad_sigmaxxout_tensor = grad_sigmaxxout.flat<double>().data();
    auto grad_sigmayyout_tensor = grad_sigmayyout.flat<double>().data();
    auto grad_sigmaxyout_tensor = grad_sigmaxyout.flat<double>().data();
    auto grad_vxout_tensor = grad_vxout.flat<double>().data();
    auto grad_vyout_tensor = grad_vyout.flat<double>().data();
    auto sigmaxxout_tensor = sigmaxxout.flat<double>().data();
    auto sigmayyout_tensor = sigmayyout.flat<double>().data();
    auto sigmaxyout_tensor = sigmaxyout.flat<double>().data();
    auto vxout_tensor = vxout.flat<double>().data();
    auto vyout_tensor = vyout.flat<double>().data();
    auto grad_sigmaxx_tensor = grad_sigmaxx->flat<double>().data();
    auto grad_sigmayy_tensor = grad_sigmayy->flat<double>().data();
    auto grad_sigmaxy_tensor = grad_sigmaxy->flat<double>().data();
    auto grad_vx_tensor = grad_vx->flat<double>().data();
    auto grad_vy_tensor = grad_vy->flat<double>().data();
    auto grad_srci_tensor = grad_srci->flat<int64>().data();
    auto grad_srcj_tensor = grad_srcj->flat<int64>().data();
    auto grad_srctype_tensor = grad_srctype->flat<int64>().data();
    auto grad_nx_tensor = grad_nx->flat<int64>().data();
    auto grad_ny_tensor = grad_ny->flat<int64>().data();
    auto grad_srcv_tensor = grad_srcv->flat<double>().data();   

    // implement your backward function here 

    // TODO:
    int64 nsrc = srcv_shape.dim_size(0);
    int64 NX = *nx.flat<int64>().data();
    int64 NY = *ny.flat<int64>().data();
    backwardCPU(
      grad_vx_tensor,
      grad_vy_tensor,
      grad_sigmaxx_tensor,
      grad_sigmayy_tensor,
      grad_sigmaxy_tensor,
      grad_srcv_tensor,
      grad_vxout_tensor,
      grad_vyout_tensor,
      grad_sigmaxxout_tensor,
      grad_sigmayyout_tensor,
      grad_sigmaxyout_tensor,
      srci_tensor, 
      srcj_tensor,
      srcv_tensor, 
      srctype_tensor,
      nsrc, NX, NY
    );
  

  }
};
REGISTER_KERNEL_BUILDER(Name("AddSourceGrad").Device(DEVICE_CPU), AddSourceGradOp);
