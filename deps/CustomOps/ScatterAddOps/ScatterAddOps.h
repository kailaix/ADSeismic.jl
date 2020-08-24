void ScatterAddOps_forward(double *out, const double *ipt, const int64 *ii,
   const double *update, int d, int n){
    for(int i = 0; i<d; i++) out[i] = ipt[i];
    for(int i = 0; i<n; i++){
      out[ii[i]-1] += update[i];      
    }
}

void ScatterAddOps_backward(
  double *grad_ipt, double *grad_update, 
  const double *grad_out,
    const double *out, const double *ipt, const int64 *ii,
   const double *update, int d, int n){
for(int i = 0; i<d; i++) grad_ipt[i] += grad_out[i];
    for(int i = 0; i<n; i++){
      grad_update[i] = grad_out[ii[i]-1];
    }
  }

