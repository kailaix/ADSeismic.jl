#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"
#include<cmath>

// Signatures for GPU kernels here 


using namespace tensorflow;
#include "AcousticOneStepCpu.h"

REGISTER_OP("AcousticOneStepCpu")
.Input("w : double")
.Input("wold : double")
.Input("phi : double")
.Input("psi : double")
.Input("sigma : double")
.Input("tau : double")
.Input("c : double")
.Input("dt : double")
.Input("hx : double")
.Input("hy : double")
.Input("nx : int64")
.Input("ny : int64")
.Output("uout : double")
.Output("phiout : double")
.Output("psiout : double")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    
        shape_inference::ShapeHandle w_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &w_shape));
        shape_inference::ShapeHandle wold_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &wold_shape));
        shape_inference::ShapeHandle phi_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &phi_shape));
        shape_inference::ShapeHandle psi_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 1, &psi_shape));
        shape_inference::ShapeHandle sigma_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(4), 1, &sigma_shape));
        shape_inference::ShapeHandle tau_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(5), 1, &tau_shape));
        shape_inference::ShapeHandle c_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(6), 1, &c_shape));
        shape_inference::ShapeHandle dt_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(7), 0, &dt_shape));
        shape_inference::ShapeHandle hx_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(8), 0, &hx_shape));
        shape_inference::ShapeHandle hy_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(9), 0, &hy_shape));
        shape_inference::ShapeHandle nx_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(10), 0, &nx_shape));
        shape_inference::ShapeHandle ny_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(11), 0, &ny_shape));

        c->set_output(0, c->Vector(-1));
        c->set_output(1, c->Vector(-1));
        c->set_output(2, c->Vector(-1));
    return Status::OK();
  });

REGISTER_OP("AcousticOneStepCpuGrad")
.Input("grad_uout : double")
.Input("grad_phiout : double")
.Input("grad_psiout : double")
.Input("uout : double")
.Input("phiout : double")
.Input("psiout : double")
.Input("w : double")
.Input("wold : double")
.Input("phi : double")
.Input("psi : double")
.Input("sigma : double")
.Input("tau : double")
.Input("c : double")
.Input("dt : double")
.Input("hx : double")
.Input("hy : double")
.Input("nx : int64")
.Input("ny : int64")
.Output("grad_w : double")
.Output("grad_wold : double")
.Output("grad_phi : double")
.Output("grad_psi : double")
.Output("grad_sigma : double")
.Output("grad_tau : double")
.Output("grad_c : double")
.Output("grad_dt : double")
.Output("grad_hx : double")
.Output("grad_hy : double")
.Output("grad_nx : int64")
.Output("grad_ny : int64");

/*-------------------------------------------------------------------------------------*/

class AcousticOneStepCpuOp : public OpKernel {
private:
  
public:
  explicit AcousticOneStepCpuOp(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {    
    DCHECK_EQ(12, context->num_inputs());
    
    
    const Tensor& w = context->input(0);
    const Tensor& wold = context->input(1);
    const Tensor& phi = context->input(2);
    const Tensor& psi = context->input(3);
    const Tensor& sigma = context->input(4);
    const Tensor& tau = context->input(5);
    const Tensor& c = context->input(6);
    const Tensor& dt = context->input(7);
    const Tensor& hx = context->input(8);
    const Tensor& hy = context->input(9);
    const Tensor& nx = context->input(10);
    const Tensor& ny = context->input(11);
    
    
    const TensorShape& w_shape = w.shape();
    const TensorShape& wold_shape = wold.shape();
    const TensorShape& phi_shape = phi.shape();
    const TensorShape& psi_shape = psi.shape();
    const TensorShape& sigma_shape = sigma.shape();
    const TensorShape& tau_shape = tau.shape();
    const TensorShape& c_shape = c.shape();
    const TensorShape& dt_shape = dt.shape();
    const TensorShape& hx_shape = hx.shape();
    const TensorShape& hy_shape = hy.shape();
    const TensorShape& nx_shape = nx.shape();
    const TensorShape& ny_shape = ny.shape();
    
    
    DCHECK_EQ(w_shape.dims(), 1);
    DCHECK_EQ(wold_shape.dims(), 1);
    DCHECK_EQ(phi_shape.dims(), 1);
    DCHECK_EQ(psi_shape.dims(), 1);
    DCHECK_EQ(sigma_shape.dims(), 1);
    DCHECK_EQ(tau_shape.dims(), 1);
    DCHECK_EQ(c_shape.dims(), 1);
    DCHECK_EQ(dt_shape.dims(), 0);
    DCHECK_EQ(hx_shape.dims(), 0);
    DCHECK_EQ(hy_shape.dims(), 0);
    DCHECK_EQ(nx_shape.dims(), 0);
    DCHECK_EQ(ny_shape.dims(), 0);

    // extra check
        
    // create output shape
    auto nx_tensor = nx.flat<int64>().data();
    auto ny_tensor = ny.flat<int64>().data();
    auto dt_tensor = dt.flat<double>().data();
    auto hx_tensor = hx.flat<double>().data();
    auto hy_tensor = hy.flat<double>().data();

    int64 NX = *nx_tensor, NY = *ny_tensor;
    double DT = *dt_tensor, HX = *hx_tensor, HY = *hy_tensor;
    
    TensorShape uout_shape({(NX+2)*(NY+2)});
    TensorShape phiout_shape({(NX+2)*(NY+2)});
    TensorShape psiout_shape({(NX+2)*(NY+2)});
            
    // create output tensor
    
    Tensor* uout = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, uout_shape, &uout));
    Tensor* phiout = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, phiout_shape, &phiout));
    Tensor* psiout = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(2, psiout_shape, &psiout));
    
    // get the corresponding Eigen tensors for data access
    
    auto w_tensor = w.flat<double>().data();
    auto wold_tensor = wold.flat<double>().data();
    auto phi_tensor = phi.flat<double>().data();
    auto psi_tensor = psi.flat<double>().data();
    auto sigma_tensor = sigma.flat<double>().data();
    auto tau_tensor = tau.flat<double>().data();
    auto c_tensor = c.flat<double>().data();
    
    
    auto uout_tensor = uout->flat<double>().data();
    auto phiout_tensor = phiout->flat<double>().data();
    auto psiout_tensor = psiout->flat<double>().data();   

    // implement your forward function here 

    // TODO:

    AcousticOneStepCpuForward(w_tensor,
                            wold_tensor,
                            phi_tensor,
                            psi_tensor,
                            sigma_tensor,
                            tau_tensor,
                            c_tensor,
                            DT, 
                            HX, HY,
                            NX, NY,
                            uout_tensor,
                            phiout_tensor,
                            psiout_tensor);

  }
};
REGISTER_KERNEL_BUILDER(Name("AcousticOneStepCpu").Device(DEVICE_CPU), AcousticOneStepCpuOp);



class AcousticOneStepCpuGradOp : public OpKernel {
private:
  
public:
  explicit AcousticOneStepCpuGradOp(OpKernelConstruction* context) : OpKernel(context) {
    
  }
  
  void Compute(OpKernelContext* context) override {
    
    
    const Tensor& grad_uout = context->input(0);
    const Tensor& grad_phiout = context->input(1);
    const Tensor& grad_psiout = context->input(2);
    const Tensor& uout = context->input(3);
    const Tensor& phiout = context->input(4);
    const Tensor& psiout = context->input(5);
    const Tensor& w = context->input(6);
    const Tensor& wold = context->input(7);
    const Tensor& phi = context->input(8);
    const Tensor& psi = context->input(9);
    const Tensor& sigma = context->input(10);
    const Tensor& tau = context->input(11);
    const Tensor& c = context->input(12);
    const Tensor& dt = context->input(13);
    const Tensor& hx = context->input(14);
    const Tensor& hy = context->input(15);
    const Tensor& nx = context->input(16);
    const Tensor& ny = context->input(17);
    
    
    const TensorShape& grad_uout_shape = grad_uout.shape();
    const TensorShape& grad_phiout_shape = grad_phiout.shape();
    const TensorShape& grad_psiout_shape = grad_psiout.shape();
    const TensorShape& uout_shape = uout.shape();
    const TensorShape& phiout_shape = phiout.shape();
    const TensorShape& psiout_shape = psiout.shape();
    const TensorShape& w_shape = w.shape();
    const TensorShape& wold_shape = wold.shape();
    const TensorShape& phi_shape = phi.shape();
    const TensorShape& psi_shape = psi.shape();
    const TensorShape& sigma_shape = sigma.shape();
    const TensorShape& tau_shape = tau.shape();
    const TensorShape& c_shape = c.shape();
    const TensorShape& dt_shape = dt.shape();
    const TensorShape& hx_shape = hx.shape();
    const TensorShape& hy_shape = hy.shape();
    const TensorShape& nx_shape = nx.shape();
    const TensorShape& ny_shape = ny.shape();
    
    
    DCHECK_EQ(grad_uout_shape.dims(), 1);
    DCHECK_EQ(grad_phiout_shape.dims(), 1);
    DCHECK_EQ(grad_psiout_shape.dims(), 1);
    DCHECK_EQ(uout_shape.dims(), 1);
    DCHECK_EQ(phiout_shape.dims(), 1);
    DCHECK_EQ(psiout_shape.dims(), 1);
    DCHECK_EQ(w_shape.dims(), 1);
    DCHECK_EQ(wold_shape.dims(), 1);
    DCHECK_EQ(phi_shape.dims(), 1);
    DCHECK_EQ(psi_shape.dims(), 1);
    DCHECK_EQ(sigma_shape.dims(), 1);
    DCHECK_EQ(tau_shape.dims(), 1);
    DCHECK_EQ(c_shape.dims(), 1);
    DCHECK_EQ(dt_shape.dims(), 0);
    DCHECK_EQ(hx_shape.dims(), 0);
    DCHECK_EQ(hy_shape.dims(), 0);
    DCHECK_EQ(nx_shape.dims(), 0);
    DCHECK_EQ(ny_shape.dims(), 0);

    // extra check
    // int m = Example.dim_size(0);
        
    // create output shape
    
    TensorShape grad_w_shape(w_shape);
    TensorShape grad_wold_shape(wold_shape);
    TensorShape grad_phi_shape(phi_shape);
    TensorShape grad_psi_shape(psi_shape);
    TensorShape grad_sigma_shape(sigma_shape);
    TensorShape grad_tau_shape(tau_shape);
    TensorShape grad_c_shape(c_shape);
    TensorShape grad_dt_shape(dt_shape);
    TensorShape grad_hx_shape(hx_shape);
    TensorShape grad_hy_shape(hy_shape);
    TensorShape grad_nx_shape(nx_shape);
    TensorShape grad_ny_shape(ny_shape);
            
    // create output tensor
    
    Tensor* grad_w = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_w_shape, &grad_w));
    Tensor* grad_wold = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, grad_wold_shape, &grad_wold));
    Tensor* grad_phi = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(2, grad_phi_shape, &grad_phi));
    Tensor* grad_psi = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(3, grad_psi_shape, &grad_psi));
    Tensor* grad_sigma = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(4, grad_sigma_shape, &grad_sigma));
    Tensor* grad_tau = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(5, grad_tau_shape, &grad_tau));
    Tensor* grad_c = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(6, grad_c_shape, &grad_c));
    Tensor* grad_dt = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(7, grad_dt_shape, &grad_dt));
    Tensor* grad_hx = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(8, grad_hx_shape, &grad_hx));
    Tensor* grad_hy = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(9, grad_hy_shape, &grad_hy));
    Tensor* grad_nx = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(10, grad_nx_shape, &grad_nx));
    Tensor* grad_ny = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(11, grad_ny_shape, &grad_ny));
    
    // get the corresponding Eigen tensors for data access
    
    auto w_tensor = w.flat<double>().data();
    auto wold_tensor = wold.flat<double>().data();
    auto phi_tensor = phi.flat<double>().data();
    auto psi_tensor = psi.flat<double>().data();
    auto sigma_tensor = sigma.flat<double>().data();
    auto tau_tensor = tau.flat<double>().data();
    auto c_tensor = c.flat<double>().data();
    auto dt_tensor = dt.flat<double>().data();
    auto hx_tensor = hx.flat<double>().data();
    auto hy_tensor = hy.flat<double>().data();
    auto nx_tensor = nx.flat<int64>().data();
    auto ny_tensor = ny.flat<int64>().data();
    auto grad_uout_tensor = grad_uout.flat<double>().data();
    auto grad_phiout_tensor = grad_phiout.flat<double>().data();
    auto grad_psiout_tensor = grad_psiout.flat<double>().data();
    auto uout_tensor = uout.flat<double>().data();
    auto phiout_tensor = phiout.flat<double>().data();
    auto psiout_tensor = psiout.flat<double>().data();
    auto grad_w_tensor = grad_w->flat<double>().data();
    auto grad_wold_tensor = grad_wold->flat<double>().data();
    auto grad_phi_tensor = grad_phi->flat<double>().data();
    auto grad_psi_tensor = grad_psi->flat<double>().data();
    auto grad_sigma_tensor = grad_sigma->flat<double>().data();
    auto grad_tau_tensor = grad_tau->flat<double>().data();
    auto grad_c_tensor = grad_c->flat<double>().data();
    auto grad_dt_tensor = grad_dt->flat<double>().data();
    auto grad_hx_tensor = grad_hx->flat<double>().data();
    auto grad_hy_tensor = grad_hy->flat<double>().data();   

    // implement your backward function here 

    // TODO:

    grad_w->flat<double>().setZero();
    grad_wold->flat<double>().setZero();
    grad_phi->flat<double>().setZero();
    grad_psi->flat<double>().setZero();
    grad_c->flat<double>().setZero();

    int64 NX = *nx_tensor, NY = *ny_tensor;
    double DT = *dt_tensor, HX = *hx_tensor, HY = *hy_tensor;
    
   
    AcousticOneStepCpuBackward(
        grad_w_tensor,
        grad_wold_tensor,
        grad_phi_tensor,
        grad_psi_tensor,
        grad_c_tensor,
        grad_uout_tensor,
        grad_phiout_tensor,
        grad_psiout_tensor,
        w_tensor,
        wold_tensor,
        phi_tensor,
        psi_tensor,
        sigma_tensor,
        tau_tensor,
        c_tensor,
        DT,
        HX,
        HY,
        NX,
        NY,
        uout_tensor,
        phiout_tensor,
        psiout_tensor);
    
  }
};
REGISTER_KERNEL_BUILDER(Name("AcousticOneStepCpuGrad").Device(DEVICE_CPU), AcousticOneStepCpuGradOp);

