#ENV["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
#using Revise
using ADSeismic
using ADCME
using PyPlot
using MAT
using DelimitedFiles
matplotlib.use("Agg")
close("all")

gpu = has_gpu() ? true : false

output_dir = "data/acoustic"
if !ispath(output_dir)
  mkpath(output_dir)
end

################### Generate synthetic data #####################
reset_default_graph()
model_name = "models/marmousi2-model-true-large.mat"
# model_name = "models/BP-model-true.mat"

## load model setting
params = load_params(model_name, vp_ref=3e3, PropagatorKernel=1)
src = load_acoustic_source(model_name)
rcv = load_acoustic_receiver(model_name)
vp = constant(matread(model_name)["vp"])
## For debug, only run the first source
# src = [src[1]]
# rcv = [rcv[1]]

## assemble acoustic propagator model
model = x->AcousticPropagatorSolver(params, x, vp^2)

## simulated wavefield 
if gpu
  Rs_ = compute_forward_GPU(model, src, rcv)
else
  [SimulatedObservation!(model(src[i]), rcv[i]) for i = 1:length(src)]
  Rs_ = [rcv[i].rcvv for i = 1:length(rcv)]
end
sess = Session(); init(sess)
Rs = run(sess, Rs_)

## save results
for i = 1:length(src)
    writedlm(joinpath(output_dir, "marmousi-r$i.txt"), Rs[i])
    # writedlm(joinpath(output_dir, "BP-r$i.txt"), Rs[i])
end
@info "Data written."

## visualize_wavefield if needed
ii = 1 # source number
u = run(sess, model(src[ii]).u)
p = visualize_wavefield(u, params)
saveanim(p, "marmousi-wavefield.gif")
