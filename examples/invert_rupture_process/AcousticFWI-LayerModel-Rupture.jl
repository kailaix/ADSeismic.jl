using ADSeismic
using ADCME
using PyPlot
using PyCall
using DelimitedFiles
using MAT
using Statistics
np = pyimport("numpy")
matplotlib.use("Agg")
use_gpu()

output_dir = "data/acoustic"
if !ispath(output_dir)
  mkpath(output_dir)
end
figure_dir = "figure/acoustic/rupture/"
if !ispath(figure_dir)
  mkpath(figure_dir)
end
result_dir = "result/acoustic/rupture/"
if !ispath(result_dir)
  mkpath(result_dir)
end
# if isfile(joinpath(result_dir, "loss.txt"))
#   rm(joinpath(result_dir, "loss.txt"))
# end

################### Generate synthetic data #####################
reset_default_graph()

ap_sim = load_acoustic_model("models/layer-model-rupture.mat"; IT_DISPLAY=0)
src = load_acoustic_source("models/layer-model-rupture.mat")
rcv_sim = load_acoustic_receiver("models/layer-model-rupture.mat")
Rs_ = compute_forward_GPU(ap_sim, src, rcv_sim)

source_scale = maximum(src[1].srcv)
vmax = maximum(src[1].srcv)
# vmin = minimum(src[1].srcv)
vmin = -vmax

sess = Session(); init(sess)
Rs = run(sess, Rs_)

for i = 1:length(src)
    writedlm(joinpath(output_dir, "rupture-r$i.txt"), Rs[i])
end

u = run(sess, ap_sim(src[1]).u)
visualize_file(u, ap_sim.param, dirs=figure_dir)

################### Inversion using Automatic Differentiation #####################
reset_default_graph()

ap_sim = load_acoustic_model("models/layer-model-rupture.mat"; IT_DISPLAY=0)
rcv_sim = load_acoustic_receiver("models/layer-model-rupture.mat")

ix = matread("models/layer-model-rupture.mat")["rupture"]["ix"][:]
iy = matread("models/layer-model-rupture.mat")["rupture"]["iy"][:]
src_ = [variable_source(ap_sim.param, ix, iy, source_scale=source_scale)]

Rs = Array{Array{Float64,2}}(undef, length(src_))
for i = 1:length(src_)
    Rs[i] = readdlm(joinpath(output_dir, "rupture-r$i.txt"))
end

sess = Session(); init(sess)

vars = [get("src")]
losses, gs = compute_loss_and_grads_GPU(ap_sim, src_, rcv_sim, Rs, vars)
g = Array{PyObject}(undef, length(gs[1]))
for i = 1:length(gs[1])
  g[i] = sum([pp[i] for pp in gs])
end

loss = sum(losses)
sess = Session(); init(sess)

function callback(v, iter, loss)
    if iter%10==0
      V = Array(reshape(v[1],  ap_sim.param.NSTEP+1, length(ix)))*source_scale
      close("all")
      t = collect(0:ap_sim.param.NSTEP)* ap_sim.param.DELTAT
      n = collect(1:length(ix)).-0.5
      n, t = np.meshgrid(n, t)
      pcolormesh(n, t, V, vmax=vmax, vmin=vmin, cmap="seismic")
      xlabel("Location index")
      ylabel("Time (s)")
      savefig(joinpath(figure_dir, "inv_$(lpad(iter,4,"0")).png"))
      writedlm(joinpath(result_dir, "inv_$(lpad(iter,5,"0")).txt"), V)
      open(joinpath(result_dir, "loss.txt"), "a") do io 
        writedlm(io, loss)
      end
    else
      return 
    end
end

callback(run(sess, vars), 0, run(sess, loss))
LBFGS!(sess, loss, vars=vars, callback=callback)
# LBFGS!(sess, loss, g, vars, callback=callback)