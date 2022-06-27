using ADSeismic
using ADCME
using PyPlot
using PyCall
using DelimitedFiles
using Statistics
use_gpu()

function mkpath_by_force(path)
  if !isdir(path)
    mkpath(path)
  end
end

output_dir = "data/acoustic"
figure_dir = "figure/acoustic/layer-model/"
result_dir = "result/acoustic/layer-model/"

mkpath_by_force(output_dir)
mkpath_by_force(figure_dir)
mkpath_by_force(result_dir)

################### Generate synthetic data #####################
reset_default_graph()

ap_sim = load_acoustic_model("models/layer-model-anomaly.mat"; IT_DISPLAY=0)
src = load_acoustic_source("models/layer-model-anomaly.mat")
rcv_sim = load_acoustic_receiver("models/layer-model-anomaly.mat")

Rs_ = compute_forward_GPU(ap_sim, src, rcv_sim)

sess = Session(); init(sess)
Rs = run(sess, Rs_)

for i = 1:length(src)
    writedlm(joinpath(output_dir, "layermodel-r$i.txt"), Rs[i])
end

u = run(sess, ap_sim(src[div(length(src),2)+1]).u)
visualize_file(u, ap_sim.param, dirs=figure_dir)

################### Inversion using Automatic Differentiation #####################
reset_default_graph()

ap_sim = load_acoustic_model("models/layer-model-smooth.mat"; inv_vp=true, IT_DISPLAY=0)
src = load_acoustic_source("models/layer-model-smooth.mat")
rcv_sim = load_acoustic_receiver("models/layer-model-smooth.mat")

Rs = Array{Array{Float64,2}}(undef, length(src))
for i = 1:length(src)
    Rs[i] = readdlm(joinpath(output_dir, "layermodel-r$i.txt"))
end

losses, gs = compute_loss_and_grads_GPU(ap_sim, src, rcv_sim, Rs, get_collection()[1])
g = sum(gs)
loss = sum(losses)
sess = Session(); init(sess)

vp = (run(sess, ap_sim.vp.contents)).^0.5
scale = mean(vp)
vmax = maximum(vp)
vmin = minimum(vp)

function callback(x, iter, loss)
  if iter%10==0
    X = Array(reshape(x, ap_sim.param.NY+2, ap_sim.param.NX+2))
    close("all")
    pcolormesh([0:ap_sim.param.NX+1;]*ap_sim.param.DELTAX,[0:ap_sim.param.NY+1;]*ap_sim.param.DELTAY,  X*scale, vmax=vmax, vmin=vmin)
    axis("scaled")
    colorbar(shrink=0.4)
    xlabel("x (km)")
    ylabel("z (km)")
    gca().invert_yaxis()
    title("Iteration = $iter")
    savefig(joinpath(figure_dir, "inv_$(lpad(iter,5,"0")).png"))
    writedlm(joinpath(result_dir, "inv_$(lpad(iter,5,"0")).txt"), X*scale)
    open(joinpath(result_dir, "loss.txt"), "a") do io 
      writedlm(io, loss)
    end
  else
    return 
  end
end

# callback(run(sess, get_collection()[1])', 0, 0)
LBFGS!(sess, loss, g, get_collection()[1]; callback=callback)

