using ADSeismic
using ADCME 
using PyPlot
using MAT
using DelimitedFiles
matplotlib.use("Agg")
close("all")

################### Generate synthetic data #####################
model_name = "../../../examples/nn_fwi/models/marmousi2-model-true.mat"
# model_name = "models/BP-model-true.mat"

## load model setting
PropagatorKernel = 1
params = load_params(model_name, vp_ref=1000, PropagatorKernel=PropagatorKernel)
src = load_acoustic_source(model_name)
rcv = load_acoustic_receiver(model_name)
vp = constant(matread(model_name)["vp"])
## For debug, only run the first source
# src = [src[1]]
# rcv = [rcv[1]]

## assemble acoustic propagator model
model = x->AcousticPropagatorSolver(params, x, vp)

## simulated wavefield 
Rs_ = compute_forward_GPU(model, src, rcv)
sess = Session(); init(sess)

# ts = zeros(11)
# for i = 1:11
#     d = @timed Rs = run(sess, Rs_)
#     ts[i] = d[2]
#     @info "Test $i, time = $(ts[i])"
# end
# @info mean(ts[2:end])

## visualize_wavefield if needed
ii = 1 # source number
u = run(sess, model(src[ii]).u)
p = visualize_wavefield(u, params)
saveanim(p,"test$PropagatorKernel.gif")