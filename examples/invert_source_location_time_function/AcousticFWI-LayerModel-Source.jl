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
figure_dir = "figure/acoustic/source/"
if !ispath(figure_dir)
  mkpath(figure_dir)
end
result_dir = "result/acoustic/source/"
if !ispath(result_dir)
  mkpath(result_dir)
end
if isfile(joinpath(result_dir, "location.txt"))
  rm(joinpath(result_dir, "location.txt"))
end
# if isfile(joinpath(result_dir, "loss.txt"))
#   rm(joinpath(result_dir, "loss.txt"))
# end

################### Generate synthetic data #####################
reset_default_graph()

ap_sim = load_acoustic_model("models/layer-model-source.mat"; IT_DISPLAY=0)
src = load_acoustic_source("models/layer-model-source.mat")
rcv_sim = load_acoustic_receiver("models/layer-model-source.mat")
Rs_ = compute_forward_GPU(ap_sim, src, rcv_sim)
source_scale = maximum(src[1].srcv)

sess = Session(); init(sess)
Rs = run(sess, Rs_)

for i = 1:length(src)
    writedlm(joinpath(output_dir, "source-r$i.txt"), Rs[i])
end

u = run(sess, ap_sim(src[1]).u)
visualize_file(u, ap_sim.param, dirs=figure_dir)

################### Inversion using Automatic Differentiation #####################
reset_default_graph()

ap_sim = load_acoustic_model("models/layer-model-source.mat"; IT_DISPLAY=0)
src = load_acoustic_source("models/layer-model-source.mat")
rcv_sim = load_acoustic_receiver("models/layer-model-source.mat")

# unknows
x_ = Variable(1.0, name="x")
y_ = Variable(0.0, name="y")
x = sigmoid(x_) * ap_sim.param.DELTAX * ap_sim.param.NX
y = sigmoid(y_) * ap_sim.param.DELTAY * ap_sim.param.NY
# x = Variable(src[1].srci * ap_sim.param.DELTAX, name="x")
# y = Variable(src[1].srcj * ap_sim.param.DELTAY, name="y")
# v = src[1].srcv
v = Variable(zeros(ap_sim.param.NSTEP), name="v") * source_scale
src_ = [AcousticSource(variable_source(ap_sim.param, x, y, v; sigma=0.5)...)]
# src_ = src

Rs = Array{Array{Float64,2}}(undef, length(src_))
for i = 1:length(src_)
    Rs[i] = readdlm(joinpath(output_dir, "source-r$i.txt"))
end

# vars = [get("x"), get("y")]
vars = [get("x"), get("y"), get("v")]
# vars = get_collection()

losses, gs = compute_loss_and_grads_GPU(ap_sim, src_, rcv_sim, Rs, vars)
g = Array{PyObject}(undef, length(gs[1]))
for i = 1:length(gs[1])
  g[i] = sum([pp[i] for pp in gs])
end
loss = sum(losses)
sess = Session(); init(sess)

@info run(sess, [loss, x, y])
@info "Exact = ", ap_sim.param.DELTAX * (src[1].srci[1]), ap_sim.param.DELTAY * (src[1].srcj[1])

function sigmoid_(x)
  return 1 / (1 + exp(-x))
end

function callback(x, iter, loss)
    if iter%4==0
      x_ = sigmoid_(x[1]) * ap_sim.param.DELTAX * ap_sim.param.NX
      y_ = sigmoid_(x[2]) * ap_sim.param.DELTAY * ap_sim.param.NY
      v_ = x[3]
      # R = x[4]
      close("all")
      figure(figsize=(10, 5))
      subplot(121)
      plot(collect(0:ap_sim.param.NSTEP-1)* ap_sim.param.DELTAT, src[1].srcv, label="Ground truth")
      plot(collect(0:ap_sim.param.NSTEP-1)* ap_sim.param.DELTAT, v_ * source_scale, label="Predicted")
      xlabel("Time (s)")
      ylim([minimum(src[1].srcv).*1.1, maximum(src[1].srcv).*1.1])
      ylabel("Amplitude")
      legend(loc="upper right")
      subplot(122)
      plot((src[1].srci .- 1.0) .* ap_sim.param.DELTAX, (src[1].srcj .- 1.0) .* ap_sim.param.DELTAY, "x", markersize=8, label="Ground truth")
      plot(x_, y_, ".", markersize=12, label="Predicted")
      xlim([0, ap_sim.param.DELTAX * ap_sim.param.NX])
      ylim([0, ap_sim.param.DELTAY * ap_sim.param.NY])
      gca().invert_yaxis()
      xlabel("X (km)")
      ylabel("Z (km)")
      legend(loc="upper right")
      suptitle("Iteration = $iter")
      savefig(joinpath(figure_dir, "inv_$(lpad(iter,5,"0")).png"))
      writedlm(joinpath(result_dir, "inv_$(lpad(iter,5,"0")).txt"), v_.*source_scale)
      open(joinpath(result_dir, "location.txt"), "a") do io 
        writedlm(io, [x_ y_])
      end      
      open(joinpath(result_dir, "loss.txt"), "a") do io 
        writedlm(io, loss)
      end
    else
      return 
    end
end

vars_callback = [vars...,rcv_sim[1].rcvv]
LBFGS!(sess, loss, vars=vars_callback, callback=callback)