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


################### Generate synthetic data #####################
output_dir = "data/acoustic"
if !ispath(output_dir)
  mkpath(output_dir)
end
figure_dir = "figure/acoustic/rupture-time/"
if !ispath(figure_dir)
  mkpath(figure_dir)
end
result_dir = "result/acoustic/rupture-time/"
if !ispath(result_dir)
  mkpath(result_dir)
end
# if isfile(joinpath(result_dir, "loss.txt"))
#   rm(joinpath(result_dir, "loss.txt"))
# end

################### Generate gaussian source #####################
# reset_default_graph()

ap_sim = load_acoustic_model("models/layer-model-rupture.mat"; IT_DISPLAY=0)
rcv_sim = load_acoustic_receiver("models/layer-model-rupture.mat")

ix = matread("models/layer-model-rupture.mat")["source"][1]["ix"][:]
iy = matread("models/layer-model-rupture.mat")["source"][1]["iy"][:]
s =  matread("models/layer-model-rupture.mat")["source"][1]["shift"][:] .* ap_sim.param.DELTAT
amp = matread("models/layer-model-rupture.mat")["source"][1]["amp"][:]
a = 1/pi

rs = []
for i = 1:length(s)
  gauss = Gauss(ap_sim.param, a, s[i], amp[i])
  push!(rs, gauss)
end
rs = hcat(rs...)
src = [AcousticSource(ix, iy, rs)]

sess = Session(); init(sess)

srcv = run(sess, src[1].srcv)
vmax = maximum(srcv)
vmin = -vmax
t = collect(0:ap_sim.param.NSTEP)* ap_sim.param.DELTAT
n = collect(1:length(src[1].srci)).-0.5
n, t = np.meshgrid(n, t)
close("all")
pcolormesh(n, t, srcv, vmax=vmax, vmin=vmin, cmap="seismic")
xlabel("Location index")
ylabel("Time (s)")
savefig(joinpath(figure_dir, "ground_truth.png"))
writedlm(joinpath(result_dir, "ground_truth.txt"), srcv)

################### Generate synthetic data #####################
# reset_default_graph()

ap_sim = load_acoustic_model("models/layer-model-rupture.mat"; IT_DISPLAY=0)
# src = load_acoustic_source("models/layer-model-rupture.mat")
rcv_sim = load_acoustic_receiver("models/layer-model-rupture.mat")
Rs_ = compute_forward_GPU(ap_sim, src, rcv_sim)

sess = Session(); init(sess)
Rs = run(sess, Rs_)

for i = 1:length(src)
    writedlm(joinpath(output_dir, "rupture-time-r$i.txt"), Rs[i])
end

u = run(sess, ap_sim(src[1]).u)
visualize_file(u, ap_sim.param, dirs=figure_dir)

################### Inversion using Automatic Differentiation #####################
reset_default_graph()

ap_sim = load_acoustic_model("models/layer-model-rupture.mat"; IT_DISPLAY=0)
rcv_sim = load_acoustic_receiver("models/layer-model-rupture.mat")

ix = matread("models/layer-model-rupture.mat")["rupture"]["ix"][:]
iy = matread("models/layer-model-rupture.mat")["rupture"]["iy"][:]
nsrc = length(ix)

# create variables
s_ = Variable(-1.5*ones(nsrc))
s = ap_sim.param.NSTEP*ap_sim.param.DELTAT*sigmoid(s_)
amp_ = Variable(0.1*ones(nsrc))
amp = 1e7*sigmoid(amp_)
if ! @isdefined a
  a = 1/pi
end

rs = []
for i = 1:length(s)
  gauss = Gauss(ap_sim.param, a, s[i], amp[i])
  push!(rs, gauss)
end
rs = hcat(rs...)
src_ = [AcousticSource(ix, iy, rs)]

Rs = Array{Array{Float64,2}}(undef, length(src_))
for i = 1:length(src_)
    Rs[i] = readdlm(joinpath(output_dir, "rupture-time-r$i.txt"))
end

sess = Session(); init(sess)

vars = [s_, amp_]
losses, gs = compute_loss_and_grads_GPU(ap_sim, src_, rcv_sim, Rs, vars)
g = Array{PyObject}(undef, length(gs[1]))
for i = 1:length(gs[1])
  g[i] = sum([pp[i] for pp in gs])
end

loss = sum(losses)
sess = Session(); init(sess)

function sigmoid_(x)
  return 1 / (1 + exp(-x))
end

function callback(v, iter, loss)
    if iter%5==0
      # @show sigmoid_.(v[2])*700.0
      # @show sigmoid_.(v[3])*1e7
      close("all")
      t = collect(0:ap_sim.param.NSTEP)* ap_sim.param.DELTAT
      n = collect(1:length(ix)).-0.5
      n, t = np.meshgrid(n, t)
      pcolormesh(n, t, v[1], vmax=vmax, vmin=vmin, cmap="seismic")
      xlabel("Location index")
      ylabel("Time (s)")
      # if iter>=0
      #   title("Iteration = $iter")
      # end
      savefig(joinpath(figure_dir, "inv_$(lpad(iter,4,"0")).png"))
      writedlm(joinpath(result_dir, "inv_$(lpad(iter,5,"0")).txt"), v[1])
      open(joinpath(result_dir, "loss.txt"), "a") do io 
        writedlm(io, loss)
      end
    else
      return 
    end
end

vars_ = [src_[1].srcv, vars...]
# callback(run(sess, vars_), 0, run(sess, loss))
LBFGS!(sess, loss, vars=vars_, callback=callback)