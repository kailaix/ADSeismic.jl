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
#include "ElasticWaveOp.h"

REGISTER_OP("ElasticWaveOp")

.Input("lambda : double")
  .Input("mu : double")
  .Input("rho : double")
  .Input("sigmaxx : double")
  .Input("sigmayy : double")
  .Input("sigmaxy : double")
  .Input("vx : double")
  .Input("vy : double")
  .Input("mem : double")
  .Input("ax : double")
  .Input("bx : double")
  .Input("kx : double")
  .Input("alphax : double")
  .Input("ay : double")
  .Input("by : double")
  .Input("ky : double")
  .Input("alphay : double")
  .Input("dx : double")
  .Input("dy : double")
  .Input("dt : double")
  .Input("srci : int64")
  .Input("srcj : int64")
  .Input("srctype : int64")
  .Input("srcv : double")
  .Output("sigmaxxout : double")
  .Output("sigmayyout : double")
  .Output("sigmaxyout : double")
  .Output("vxout : double")
  .Output("vyout : double")
  .Output("memout : double")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    
        shape_inference::ShapeHandle lambda_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &lambda_shape));
        shape_inference::ShapeHandle mu_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &mu_shape));
        shape_inference::ShapeHandle rho_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 2, &rho_shape));
        shape_inference::ShapeHandle sigmaxx_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 2, &sigmaxx_shape));
        shape_inference::ShapeHandle sigmayy_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(4), 2, &sigmayy_shape));
        shape_inference::ShapeHandle sigmaxy_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(5), 2, &sigmaxy_shape));
        shape_inference::ShapeHandle vx_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(6), 2, &vx_shape));
        shape_inference::ShapeHandle vy_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(7), 2, &vy_shape));
        shape_inference::ShapeHandle mem_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(8), 3, &mem_shape));
        shape_inference::ShapeHandle ax_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(9), 2, &ax_shape));
        shape_inference::ShapeHandle bx_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(10), 2, &bx_shape));
        shape_inference::ShapeHandle kx_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(11), 2, &kx_shape));
        shape_inference::ShapeHandle alphax_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(12), 2, &alphax_shape));
        shape_inference::ShapeHandle ay_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(13), 2, &ay_shape));
        shape_inference::ShapeHandle by_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(14), 2, &by_shape));
        shape_inference::ShapeHandle ky_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(15), 2, &ky_shape));
        shape_inference::ShapeHandle alphay_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(16), 2, &alphay_shape));
        shape_inference::ShapeHandle dx_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(17), 0, &dx_shape));
        shape_inference::ShapeHandle dy_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(18), 0, &dy_shape));
        shape_inference::ShapeHandle dt_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(19), 0, &dt_shape));
        shape_inference::ShapeHandle srci_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(20), 1, &srci_shape));
        shape_inference::ShapeHandle srcj_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(21), 1, &srcj_shape));
        shape_inference::ShapeHandle srctype_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(22), 1, &srctype_shape));
        shape_inference::ShapeHandle srcv_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(23), 1, &srcv_shape));

        c->set_output(0, c->input(3));
        c->set_output(1, c->input(3));
        c->set_output(2, c->input(3));
        c->set_output(3, c->input(3));
        c->set_output(4, c->input(3));
        c->set_output(5, c->input(8));
    return Status::OK();
  });
class ElasticWaveOpOp : public OpKernel {
private:
  
public:
  explicit ElasticWaveOpOp(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {    
    DCHECK_EQ(24, context->num_inputs());
    
    
    const Tensor& lambda = context->input(0);
    const Tensor& mu = context->input(1);
    const Tensor& rho = context->input(2);
    const Tensor& sigmaxx = context->input(3);
    const Tensor& sigmayy = context->input(4);
    const Tensor& sigmaxy = context->input(5);
    const Tensor& vx = context->input(6);
    const Tensor& vy = context->input(7);
    const Tensor& mem = context->input(8);
    const Tensor& ax = context->input(9);
    const Tensor& bx = context->input(10);
    const Tensor& kx = context->input(11);
    const Tensor& alphax = context->input(12);
    const Tensor& ay = context->input(13);
    const Tensor& by = context->input(14);
    const Tensor& ky = context->input(15);
    const Tensor& alphay = context->input(16);
    const Tensor& dx = context->input(17);
    const Tensor& dy = context->input(18);
    const Tensor& dt = context->input(19);
    const Tensor& srci = context->input(20);
    const Tensor& srcj = context->input(21);
    const Tensor& srctype = context->input(22);
    const Tensor& srcv = context->input(23);
    
    
    const TensorShape& lambda_shape = lambda.shape();
    const TensorShape& mu_shape = mu.shape();
    const TensorShape& rho_shape = rho.shape();
    const TensorShape& sigmaxx_shape = sigmaxx.shape();
    const TensorShape& sigmayy_shape = sigmayy.shape();
    const TensorShape& sigmaxy_shape = sigmaxy.shape();
    const TensorShape& vx_shape = vx.shape();
    const TensorShape& vy_shape = vy.shape();
    const TensorShape& mem_shape = mem.shape();
    const TensorShape& ax_shape = ax.shape();
    const TensorShape& bx_shape = bx.shape();
    const TensorShape& kx_shape = kx.shape();
    const TensorShape& alphax_shape = alphax.shape();
    const TensorShape& ay_shape = ay.shape();
    const TensorShape& by_shape = by.shape();
    const TensorShape& ky_shape = ky.shape();
    const TensorShape& alphay_shape = alphay.shape();
    const TensorShape& dx_shape = dx.shape();
    const TensorShape& dy_shape = dy.shape();
    const TensorShape& dt_shape = dt.shape();
    const TensorShape& srci_shape = srci.shape();
    const TensorShape& srcj_shape = srcj.shape();
    const TensorShape& srctype_shape = srctype.shape();
    const TensorShape& srcv_shape = srcv.shape();
    
    
    DCHECK_EQ(lambda_shape.dims(), 2);
    DCHECK_EQ(mu_shape.dims(), 2);
    DCHECK_EQ(rho_shape.dims(), 2);
    DCHECK_EQ(sigmaxx_shape.dims(), 2);
    DCHECK_EQ(sigmayy_shape.dims(), 2);
    DCHECK_EQ(sigmaxy_shape.dims(), 2);
    DCHECK_EQ(vx_shape.dims(), 2);
    DCHECK_EQ(vy_shape.dims(), 2);
    DCHECK_EQ(mem_shape.dims(), 3);
    DCHECK_EQ(ax_shape.dims(), 2);
    DCHECK_EQ(bx_shape.dims(), 2);
    DCHECK_EQ(kx_shape.dims(), 2);
    DCHECK_EQ(alphax_shape.dims(), 2);
    DCHECK_EQ(ay_shape.dims(), 2);
    DCHECK_EQ(by_shape.dims(), 2);
    DCHECK_EQ(ky_shape.dims(), 2);
    DCHECK_EQ(alphay_shape.dims(), 2);
    DCHECK_EQ(dx_shape.dims(), 0);
    DCHECK_EQ(dy_shape.dims(), 0);
    DCHECK_EQ(dt_shape.dims(), 0);
    DCHECK_EQ(srci_shape.dims(), 1);
    DCHECK_EQ(srcj_shape.dims(), 1);
    DCHECK_EQ(srctype_shape.dims(), 1);
    DCHECK_EQ(srcv_shape.dims(), 1);

    // extra check
        
    // create output shape
    int64 NX = sigmaxx_shape.dim_size(0)-2;
    int64 NY = sigmayy_shape.dim_size(1)-2;
    int64 N = (NX+2)*(NY+2);
    int64 nsrc = srci_shape.dim_size(0);
    TensorShape sigmaxxout_shape({NX+2, NY+2});
    TensorShape sigmayyout_shape({NX+2, NY+2});
    TensorShape sigmaxyout_shape({NX+2, NY+2});
    TensorShape vxout_shape({NX+2, NY+2});
    TensorShape vyout_shape({NX+2, NY+2});
    TensorShape memout_shape({8,NX+2, NY+2});
            
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
    Tensor* memout = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(5, memout_shape, &memout));
    
    // get the corresponding Eigen tensors for data access
    
    auto lambda_tensor = lambda.flat<double>().data();
    auto mu_tensor = mu.flat<double>().data();
    auto rho_tensor = rho.flat<double>().data();
    auto sigmaxx_tensor = sigmaxx.flat<double>().data();
    auto sigmayy_tensor = sigmayy.flat<double>().data();
    auto sigmaxy_tensor = sigmaxy.flat<double>().data();
    auto vx_tensor = vx.flat<double>().data();
    auto vy_tensor = vy.flat<double>().data();
    auto mem_tensor = mem.flat<double>().data();
    auto ax_tensor = ax.flat<double>().data();
    auto bx_tensor = bx.flat<double>().data();
    auto kx_tensor = kx.flat<double>().data();
    auto alphax_tensor = alphax.flat<double>().data();
    auto ay_tensor = ay.flat<double>().data();
    auto by_tensor = by.flat<double>().data();
    auto ky_tensor = ky.flat<double>().data();
    auto alphay_tensor = alphay.flat<double>().data();
    auto dx_tensor = dx.flat<double>().data();
    auto dy_tensor = dy.flat<double>().data();
    auto dt_tensor = dt.flat<double>().data();
    auto srci_tensor = srci.flat<int64>().data();
    auto srcj_tensor = srcj.flat<int64>().data();
    auto srctype_tensor = srctype.flat<int64>().data();
    auto srcv_tensor = srcv.flat<double>().data();
    auto sigmaxxout_tensor = sigmaxxout->flat<double>().data();
    auto sigmayyout_tensor = sigmayyout->flat<double>().data();
    auto sigmaxyout_tensor = sigmaxyout->flat<double>().data();
    auto vxout_tensor = vxout->flat<double>().data();
    auto vyout_tensor = vyout->flat<double>().data();
    auto memout_tensor = memout->flat<double>().data();   

    // implement your forward function here 

    // TODO:
    forward(
        vxout_tensor,              // double * vx_,
        vyout_tensor,              // double * vy_,
        sigmaxxout_tensor,              // double * sigmaxx_,
        sigmayyout_tensor,              // double * sigmayy_,
        sigmaxyout_tensor,              // double * sigmaxy_,
        lambda_tensor,              // const double * lambda,
        mu_tensor,              // const double * mu,
        rho_tensor,              // const double * rho,
        memout_tensor,              // double * memory_dvx_dx_,
        memout_tensor+N,              // double * memory_dvx_dy_,
        memout_tensor+2*N,              // double * memory_dvy_dx_,
        memout_tensor+3*N,              // double * memory_dvy_dy_,
        memout_tensor+4*N,              // double * memory_dsigmaxx_dx_,
        memout_tensor+5*N,              // double * memory_dsigmayy_dy_,
        memout_tensor+6*N,              // double * memory_dsigmaxy_dx_,
        memout_tensor+7*N,              // double * memory_dsigmaxy_dy_,
        vx_tensor,              // const double * vx,
        vy_tensor,             // const double * vy,
        sigmaxx_tensor,              // const double * sigmaxx,
        sigmayy_tensor,              // const double * sigmayy,
        sigmaxy_tensor,              // const double * sigmaxy,
        mem_tensor,                  // const double * memory_dvx_dx,
        mem_tensor+N,                // const double * memory_dvx_dy,
        mem_tensor+2*N,              // const double * memory_dvy_dx,
        mem_tensor+3*N,              // const double * memory_dvy_dy,
        mem_tensor+4*N,              // const double * memory_dsigmaxx_dx,
        mem_tensor+5*N,              // const double * memory_dsigmayy_dy,
        mem_tensor+6*N,              // const double * memory_dsigmaxy_dx,
        mem_tensor+7*N,              // const double * memory_dsigmaxy_dy,
        kx_tensor,              // const double * K_x,   // PML parameter
        ky_tensor,              // const double * K_y,
        kx_tensor+NX,              // const double * K_x_half,
        ky_tensor+NY,              // const double * K_y_half,
        bx_tensor,              // const double * b_x,
        by_tensor,              // const double * b_y,
        bx_tensor+NX,              // const double * b_x_half,
        by_tensor+NY,              // const double * b_y_half,
        ax_tensor,              // const double * a_x,
        ay_tensor,              // const double * a_y,
        ax_tensor+NX,              // const double * a_x_half,
        ay_tensor+NY,              // const double * a_y_half,
        alphax_tensor,              // const double * alpha_x,
        alphay_tensor,              // const double * alpha_y,
        alphax_tensor+NX,              // const double * alpha_x_half,
        alphay_tensor+NY,              // const double * alpha_y_half,
        srci_tensor, srcj_tensor, srcv_tensor, srctype_tensor, nsrc,             // const int64 * srci, const int64 *srcj, const double *srcv, const int64 *src_type, int64 nsrc,
        *dx_tensor, *dy_tensor, *dt_tensor,              // double DELTAX, double DELTAY, double DELTAT,
        NX, NY              // int64 NX, int64 NY,
    );

  }
};
REGISTER_KERNEL_BUILDER(Name("ElasticWaveOp").Device(DEVICE_CPU), ElasticWaveOpOp);


REGISTER_OP("ElasticWaveOpGrad")
  
  .Input("grad_sigmaxxout : double")
.Input("grad_sigmayyout : double")
.Input("grad_sigmaxyout : double")
.Input("grad_vxout : double")
.Input("grad_vyout : double")
.Input("grad_memout : double")
  .Input("sigmaxxout : double")
  .Input("sigmayyout : double")
  .Input("sigmaxyout : double")
  .Input("vxout : double")
  .Input("vyout : double")
  .Input("memout : double")
  .Input("lambda : double")
  .Input("mu : double")
  .Input("rho : double")
  .Input("sigmaxx : double")
  .Input("sigmayy : double")
  .Input("sigmaxy : double")
  .Input("vx : double")
  .Input("vy : double")
  .Input("mem : double")
  .Input("ax : double")
  .Input("bx : double")
  .Input("kx : double")
  .Input("alphax : double")
  .Input("ay : double")
  .Input("by : double")
  .Input("ky : double")
  .Input("alphay : double")
  .Input("dx : double")
  .Input("dy : double")
  .Input("dt : double")
  .Input("srci : int64")
  .Input("srcj : int64")
  .Input("srctype : int64")
  .Input("srcv : double")
  .Output("grad_lambda : double")
  .Output("grad_mu : double")
  .Output("grad_rho : double")
  .Output("grad_sigmaxx : double")
  .Output("grad_sigmayy : double")
  .Output("grad_sigmaxy : double")
  .Output("grad_vx : double")
  .Output("grad_vy : double")
  .Output("grad_mem : double")
  .Output("grad_ax : double")
  .Output("grad_bx : double")
  .Output("grad_kx : double")
  .Output("grad_alphax : double")
  .Output("grad_ay : double")
  .Output("grad_by : double")
  .Output("grad_ky : double")
  .Output("grad_alphay : double")
  .Output("grad_dx : double")
  .Output("grad_dy : double")
  .Output("grad_dt : double")
  .Output("grad_srci : int64")
  .Output("grad_srcj : int64")
  .Output("grad_srctype : int64")
  .Output("grad_srcv : double");
class ElasticWaveOpGradOp : public OpKernel {
private:
  
public:
  explicit ElasticWaveOpGradOp(OpKernelConstruction* context) : OpKernel(context) {
    
  }
  
  void Compute(OpKernelContext* context) override {
    
    
    const Tensor& grad_sigmaxxout = context->input(0);
    const Tensor& grad_sigmayyout = context->input(1);
    const Tensor& grad_sigmaxyout = context->input(2);
    const Tensor& grad_vxout = context->input(3);
    const Tensor& grad_vyout = context->input(4);
    const Tensor& grad_memout = context->input(5);
    const Tensor& sigmaxxout = context->input(6);
    const Tensor& sigmayyout = context->input(7);
    const Tensor& sigmaxyout = context->input(8);
    const Tensor& vxout = context->input(9);
    const Tensor& vyout = context->input(10);
    const Tensor& memout = context->input(11);
    const Tensor& lambda = context->input(12);
    const Tensor& mu = context->input(13);
    const Tensor& rho = context->input(14);
    const Tensor& sigmaxx = context->input(15);
    const Tensor& sigmayy = context->input(16);
    const Tensor& sigmaxy = context->input(17);
    const Tensor& vx = context->input(18);
    const Tensor& vy = context->input(19);
    const Tensor& mem = context->input(20);
    const Tensor& ax = context->input(21);
    const Tensor& bx = context->input(22);
    const Tensor& kx = context->input(23);
    const Tensor& alphax = context->input(24);
    const Tensor& ay = context->input(25);
    const Tensor& by = context->input(26);
    const Tensor& ky = context->input(27);
    const Tensor& alphay = context->input(28);
    const Tensor& dx = context->input(29);
    const Tensor& dy = context->input(30);
    const Tensor& dt = context->input(31);
    const Tensor& srci = context->input(32);
    const Tensor& srcj = context->input(33);
    const Tensor& srctype = context->input(34);
    const Tensor& srcv = context->input(35);
    
    
    const TensorShape& grad_sigmaxxout_shape = grad_sigmaxxout.shape();
    const TensorShape& grad_sigmayyout_shape = grad_sigmayyout.shape();
    const TensorShape& grad_sigmaxyout_shape = grad_sigmaxyout.shape();
    const TensorShape& grad_vxout_shape = grad_vxout.shape();
    const TensorShape& grad_vyout_shape = grad_vyout.shape();
    const TensorShape& grad_memout_shape = grad_memout.shape();
    const TensorShape& sigmaxxout_shape = sigmaxxout.shape();
    const TensorShape& sigmayyout_shape = sigmayyout.shape();
    const TensorShape& sigmaxyout_shape = sigmaxyout.shape();
    const TensorShape& vxout_shape = vxout.shape();
    const TensorShape& vyout_shape = vyout.shape();
    const TensorShape& memout_shape = memout.shape();
    const TensorShape& lambda_shape = lambda.shape();
    const TensorShape& mu_shape = mu.shape();
    const TensorShape& rho_shape = rho.shape();
    const TensorShape& sigmaxx_shape = sigmaxx.shape();
    const TensorShape& sigmayy_shape = sigmayy.shape();
    const TensorShape& sigmaxy_shape = sigmaxy.shape();
    const TensorShape& vx_shape = vx.shape();
    const TensorShape& vy_shape = vy.shape();
    const TensorShape& mem_shape = mem.shape();
    const TensorShape& ax_shape = ax.shape();
    const TensorShape& bx_shape = bx.shape();
    const TensorShape& kx_shape = kx.shape();
    const TensorShape& alphax_shape = alphax.shape();
    const TensorShape& ay_shape = ay.shape();
    const TensorShape& by_shape = by.shape();
    const TensorShape& ky_shape = ky.shape();
    const TensorShape& alphay_shape = alphay.shape();
    const TensorShape& dx_shape = dx.shape();
    const TensorShape& dy_shape = dy.shape();
    const TensorShape& dt_shape = dt.shape();
    const TensorShape& srci_shape = srci.shape();
    const TensorShape& srcj_shape = srcj.shape();
    const TensorShape& srctype_shape = srctype.shape();
    const TensorShape& srcv_shape = srcv.shape();
    
    
    DCHECK_EQ(grad_sigmaxxout_shape.dims(), 2);
    DCHECK_EQ(grad_sigmayyout_shape.dims(), 2);
    DCHECK_EQ(grad_sigmaxyout_shape.dims(), 2);
    DCHECK_EQ(grad_vxout_shape.dims(), 2);
    DCHECK_EQ(grad_vyout_shape.dims(), 2);
    DCHECK_EQ(grad_memout_shape.dims(), 3);
    DCHECK_EQ(sigmaxxout_shape.dims(), 2);
    DCHECK_EQ(sigmayyout_shape.dims(), 2);
    DCHECK_EQ(sigmaxyout_shape.dims(), 2);
    DCHECK_EQ(vxout_shape.dims(), 2);
    DCHECK_EQ(vyout_shape.dims(), 2);
    DCHECK_EQ(memout_shape.dims(), 3);
    DCHECK_EQ(lambda_shape.dims(), 2);
    DCHECK_EQ(mu_shape.dims(), 2);
    DCHECK_EQ(rho_shape.dims(), 2);
    DCHECK_EQ(sigmaxx_shape.dims(), 2);
    DCHECK_EQ(sigmayy_shape.dims(), 2);
    DCHECK_EQ(sigmaxy_shape.dims(), 2);
    DCHECK_EQ(vx_shape.dims(), 2);
    DCHECK_EQ(vy_shape.dims(), 2);
    DCHECK_EQ(mem_shape.dims(), 3);
    DCHECK_EQ(ax_shape.dims(), 2);
    DCHECK_EQ(bx_shape.dims(), 2);
    DCHECK_EQ(kx_shape.dims(), 2);
    DCHECK_EQ(alphax_shape.dims(), 2);
    DCHECK_EQ(ay_shape.dims(), 2);
    DCHECK_EQ(by_shape.dims(), 2);
    DCHECK_EQ(ky_shape.dims(), 2);
    DCHECK_EQ(alphay_shape.dims(), 2);
    DCHECK_EQ(dx_shape.dims(), 0);
    DCHECK_EQ(dy_shape.dims(), 0);
    DCHECK_EQ(dt_shape.dims(), 0);
    DCHECK_EQ(srci_shape.dims(), 1);
    DCHECK_EQ(srcj_shape.dims(), 1);
    DCHECK_EQ(srctype_shape.dims(), 1);
    DCHECK_EQ(srcv_shape.dims(), 1);

    // extra check
    // int m = Example.dim_size(0);
        
    // create output shape
    
    TensorShape grad_lambda_shape(lambda_shape);
    TensorShape grad_mu_shape(mu_shape);
    TensorShape grad_rho_shape(rho_shape);
    TensorShape grad_sigmaxx_shape(sigmaxx_shape);
    TensorShape grad_sigmayy_shape(sigmayy_shape);
    TensorShape grad_sigmaxy_shape(sigmaxy_shape);
    TensorShape grad_vx_shape(vx_shape);
    TensorShape grad_vy_shape(vy_shape);
    TensorShape grad_mem_shape(mem_shape);
    TensorShape grad_ax_shape(ax_shape);
    TensorShape grad_bx_shape(bx_shape);
    TensorShape grad_kx_shape(kx_shape);
    TensorShape grad_alphax_shape(alphax_shape);
    TensorShape grad_ay_shape(ay_shape);
    TensorShape grad_by_shape(by_shape);
    TensorShape grad_ky_shape(ky_shape);
    TensorShape grad_alphay_shape(alphay_shape);
    TensorShape grad_dx_shape(dx_shape);
    TensorShape grad_dy_shape(dy_shape);
    TensorShape grad_dt_shape(dt_shape);
    TensorShape grad_srci_shape(srci_shape);
    TensorShape grad_srcj_shape(srcj_shape);
    TensorShape grad_srctype_shape(srctype_shape);
    TensorShape grad_srcv_shape(srcv_shape);
            
    // create output tensor
    
    Tensor* grad_lambda = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_lambda_shape, &grad_lambda));
    Tensor* grad_mu = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, grad_mu_shape, &grad_mu));
    Tensor* grad_rho = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(2, grad_rho_shape, &grad_rho));
    Tensor* grad_sigmaxx = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(3, grad_sigmaxx_shape, &grad_sigmaxx));
    Tensor* grad_sigmayy = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(4, grad_sigmayy_shape, &grad_sigmayy));
    Tensor* grad_sigmaxy = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(5, grad_sigmaxy_shape, &grad_sigmaxy));
    Tensor* grad_vx = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(6, grad_vx_shape, &grad_vx));
    Tensor* grad_vy = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(7, grad_vy_shape, &grad_vy));
    Tensor* grad_mem = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(8, grad_mem_shape, &grad_mem));
    Tensor* grad_ax = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(9, grad_ax_shape, &grad_ax));
    Tensor* grad_bx = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(10, grad_bx_shape, &grad_bx));
    Tensor* grad_kx = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(11, grad_kx_shape, &grad_kx));
    Tensor* grad_alphax = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(12, grad_alphax_shape, &grad_alphax));
    Tensor* grad_ay = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(13, grad_ay_shape, &grad_ay));
    Tensor* grad_by = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(14, grad_by_shape, &grad_by));
    Tensor* grad_ky = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(15, grad_ky_shape, &grad_ky));
    Tensor* grad_alphay = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(16, grad_alphay_shape, &grad_alphay));
    Tensor* grad_dx = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(17, grad_dx_shape, &grad_dx));
    Tensor* grad_dy = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(18, grad_dy_shape, &grad_dy));
    Tensor* grad_dt = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(19, grad_dt_shape, &grad_dt));
    Tensor* grad_srci = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(20, grad_srci_shape, &grad_srci));
    Tensor* grad_srcj = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(21, grad_srcj_shape, &grad_srcj));
    Tensor* grad_srctype = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(22, grad_srctype_shape, &grad_srctype));
    Tensor* grad_srcv = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(23, grad_srcv_shape, &grad_srcv));
    
    // get the corresponding Eigen tensors for data access
    
    auto lambda_tensor = lambda.flat<double>().data();
    auto mu_tensor = mu.flat<double>().data();
    auto rho_tensor = rho.flat<double>().data();
    auto sigmaxx_tensor = sigmaxx.flat<double>().data();
    auto sigmayy_tensor = sigmayy.flat<double>().data();
    auto sigmaxy_tensor = sigmaxy.flat<double>().data();
    auto vx_tensor = vx.flat<double>().data();
    auto vy_tensor = vy.flat<double>().data();
    auto mem_tensor = mem.flat<double>().data();
    auto ax_tensor = ax.flat<double>().data();
    auto bx_tensor = bx.flat<double>().data();
    auto kx_tensor = kx.flat<double>().data();
    auto alphax_tensor = alphax.flat<double>().data();
    auto ay_tensor = ay.flat<double>().data();
    auto by_tensor = by.flat<double>().data();
    auto ky_tensor = ky.flat<double>().data();
    auto alphay_tensor = alphay.flat<double>().data();
    auto dx_tensor = dx.flat<double>().data();
    auto dy_tensor = dy.flat<double>().data();
    auto dt_tensor = dt.flat<double>().data();
    auto srci_tensor = srci.flat<int64>().data();
    auto srcj_tensor = srcj.flat<int64>().data();
    auto srctype_tensor = srctype.flat<int64>().data();
    auto srcv_tensor = srcv.flat<double>().data();
    auto grad_sigmaxxout_tensor = grad_sigmaxxout.flat<double>().data();
    auto grad_sigmayyout_tensor = grad_sigmayyout.flat<double>().data();
    auto grad_sigmaxyout_tensor = grad_sigmaxyout.flat<double>().data();
    auto grad_vxout_tensor = grad_vxout.flat<double>().data();
    auto grad_vyout_tensor = grad_vyout.flat<double>().data();
    auto grad_memout_tensor = grad_memout.flat<double>().data();
    auto sigmaxxout_tensor = sigmaxxout.flat<double>().data();
    auto sigmayyout_tensor = sigmayyout.flat<double>().data();
    auto sigmaxyout_tensor = sigmaxyout.flat<double>().data();
    auto vxout_tensor = vxout.flat<double>().data();
    auto vyout_tensor = vyout.flat<double>().data();
    auto memout_tensor = memout.flat<double>().data();
    auto grad_lambda_tensor = grad_lambda->flat<double>().data();
    auto grad_mu_tensor = grad_mu->flat<double>().data();
    auto grad_rho_tensor = grad_rho->flat<double>().data();
    auto grad_sigmaxx_tensor = grad_sigmaxx->flat<double>().data();
    auto grad_sigmayy_tensor = grad_sigmayy->flat<double>().data();
    auto grad_sigmaxy_tensor = grad_sigmaxy->flat<double>().data();
    auto grad_vx_tensor = grad_vx->flat<double>().data();
    auto grad_vy_tensor = grad_vy->flat<double>().data();
    auto grad_mem_tensor = grad_mem->flat<double>().data();
    auto grad_ax_tensor = grad_ax->flat<double>().data();
    auto grad_bx_tensor = grad_bx->flat<double>().data();
    auto grad_kx_tensor = grad_kx->flat<double>().data();
    auto grad_alphax_tensor = grad_alphax->flat<double>().data();
    auto grad_ay_tensor = grad_ay->flat<double>().data();
    auto grad_by_tensor = grad_by->flat<double>().data();
    auto grad_ky_tensor = grad_ky->flat<double>().data();
    auto grad_alphay_tensor = grad_alphay->flat<double>().data();
    auto grad_dx_tensor = grad_dx->flat<double>().data();
    auto grad_dy_tensor = grad_dy->flat<double>().data();
    auto grad_dt_tensor = grad_dt->flat<double>().data();
    auto grad_srci_tensor = grad_srci->flat<int64>().data();
    auto grad_srcj_tensor = grad_srcj->flat<int64>().data();
    auto grad_srctype_tensor = grad_srctype->flat<int64>().data();
    auto grad_srcv_tensor = grad_srcv->flat<double>().data();   

    // implement your backward function here 

    // TODO:
    
  }
};
REGISTER_KERNEL_BUILDER(Name("ElasticWaveOpGrad").Device(DEVICE_CPU), ElasticWaveOpGradOp);

