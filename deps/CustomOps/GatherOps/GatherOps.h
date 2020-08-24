void GatherOps_forward(double *out, const double *v, const int64 *ii, int n){
  for(int i = 0; i<n; i++){
    out[i] = v[ii[i]-1];
  }
}

void GatherOps_backward(
  double *grad_v, 
  const double *grad_out, 
  const double *out, const double *v, const int64 *ii, int n
){
  for(int i = 0; i<n; i++)
    grad_v[ii[i]-1] = grad_out[i];
}