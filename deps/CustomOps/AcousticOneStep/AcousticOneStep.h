#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"
#include<cmath>
#include <cuda_runtime_api.h>
#include "cuda.h"
using namespace tensorflow;
void AcousticOneStepForward(const double*  w,
    const double*  wold,
    const double*  phi,
    const double*  psi,
    const double*  sigma,
    const double*  tau,
    const double*  c,
    double  dt,
    double  hx,
    double  hy,
    int64  NX,
    int64  NY,
    double*  u,
    double*  phiout,
    double*  psiout);

void AcousticOneStepBackward(
    double*  grad_w,
    double*  grad_wold,
    double*  grad_phi,
    double*  grad_psi,
    double*  grad_c,
    const double*  grad_u,
    const double*  grad_phiout,
    const double*  grad_psiout,
    const double*  w,
    const double*  wold,
    const double*  phi,
    const double*  psi,
    const double*  sigma,
    const double*  tau,
    const double*  c,
    double  dt,
    double  hx,
    double  hy,
    int64  NX,
    int64  NY,
    const double*  u,
    const double*  phiout,
    const double*  psiout);