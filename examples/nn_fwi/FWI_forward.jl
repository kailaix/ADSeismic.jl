using Revise
using ADSeismic
using ADCME
using PyPlot
using MAT
using DelimitedFiles
# matplotlib.use("Agg")
close("all")
if has_gpu()
  gpu = true
else
  gpu = false
end

output_dir = "data/acoustic"
if !ispath(output_dir)
  mkpath(output_dir)
end

################### Generate synthetic data #####################
model_name = "models/marmousi2-model-true.mat"

## load model setting
params = load_params(model_name)
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
end

## visualize_wavefield if needed
ii = 1 # source number
u = run(sess, model(src[ii]).u)
visualize_wavefield(u, params)