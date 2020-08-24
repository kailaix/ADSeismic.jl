void ScatterNdOps_forward(double *out, const int64 *ii,
   const double *update, int n){
    for(int i = 0; i<n; i++){
      out[ii[i]-1] = update[i];      
    }
}

void ScatterNdOps_backward(
  double *grad_update, 
  const double *grad_out,
    const double *out, const int64 *ii,
   const double *update, int n){
    for(int i = 0; i<n; i++){
      grad_update[i] = grad_out[ii[i]-1];
    }
}

