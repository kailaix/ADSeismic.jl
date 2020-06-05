using Revise
using ADSeismic
using ADCME
using PyPlot
using DelimitedFiles
# matplotlib.use("Agg")
close("all")
if has_gpu()
  use_gpu()
  gpu = true
else
  gpu = false
end


output_dir = "data/acoustic"
if !ispath(output_dir)
  mkpath(output_dir)
end
figure_dir = "figure/acoustic/marmousi/"
if !ispath(figure_dir)
  mkpath(figure_dir)
end
result_dir = "result/acoustic/marmousi/"
if !ispath(result_dir)
  mkpath(result_dir)
end
# if isfile(joinpath(result_dir, "loss.txt"))
#   rm(joinpath(result_dir, "loss.txt"))
# end

################### Generate synthetic data #####################
reset_default_graph()

ap_sim = load_acoustic_model("models/marmousi2-model-true.mat", IT_DISPLAY=0)
src = load_acoustic_source("models/marmousi2-model-true.mat")
rcv_sim = load_acoustic_receiver("models/marmousi2-model-true.mat")
## For debug, only run the first source
# src = [src[1]]
# rcv_sim = [rcv_sim[1]]

if gpu
  Rs_ = compute_forward_GPU(ap_sim, src, rcv_sim)
else
  [SimulatedObservation!(ap_sim(src[i]), rcv_sim[i]) for i = 1:length(src)]
  Rs_ = [rcv_sim[i].rcvv for i = 1:length(rcv_sim)]
end

sess = Session(); init(sess)

Rs = run(sess, Rs_)

for i = 1:length(src)
    writedlm(joinpath(output_dir, "marmousi-r$i.txt"), Rs[i])
end

## visualize_wavefield if needed
# u = run(sess, ap_sim(src[div(length(src),2)+1]).u)
# visualize_wavefield(u, ap_sim.param)

################### Inversion using Automatic Differentiation #####################
reset_default_graph()

ap_sim = load_acoustic_model("models/marmousi2-model-true.mat"; inv_vp=true, IT_DISPLAY=0)
src = load_acoustic_source("models/marmousi2-model-true.mat")
rcv_sim = load_acoustic_receiver("models/marmousi2-model-true.mat")
## For debug, only run the first source
# src = [src[1]]
# rcv_sim = [rcv_sim[1]]

Rs = Array{Array{Float64,2}}(undef, length(src))
for i = 1:length(src)
    Rs[i] = readdlm(joinpath(output_dir, "marmousi-r$i.txt"))
end

if gpu
  losses, gs = compute_loss_and_grads_GPU(ap_sim, src, rcv_sim, Rs, get_collection()[1])
  g = sum(gs)
  loss = sum(losses)
else
  [SimulatedObservation!(ap_sim(src[i]), rcv_sim[i]) for i = 1:length(src)]
  loss = sum([sum((rcv_sim[i].rcvv-Rs[i])^2) for i = 1:length(rcv_sim)]) 
end

sess = Session(); init(sess)

@show run(sess, losses)
error()

vp = (run(sess, ap_sim.vp.contents)).^0.5./1e3
scale = mean(vp)
vmax = maximum(vp)
vmin = minimum(vp)

function callback(vs, iter, loss)
  if iter%10==0
    if ndims(vs) == 1
      vp = Array(reshape(vs[1:(ap_sim.param.NY+2)*(ap_sim.param.NX+2)], ap_sim.param.NY+2, ap_sim.param.NX+2))
    else
      vp = vs[1]'
    end
    clf()
    pcolormesh([0:ap_sim.param.NX+1;]*ap_sim.param.DELTAX/1e3,[0:ap_sim.param.NY+1;]*ap_sim.param.DELTAY/1e3,  vp*scale)
    axis("scaled")
    colorbar(shrink=0.4)
    xlabel("x (km)")
    ylabel("z (km)")
    gca().invert_yaxis()
    title("Iteration = $iter")
    savefig(joinpath(figure_dir, "inv_$(lpad(iter,5,"0")).png"), bbox_inches="tight")
    writedlm(joinpath(result_dir, "inv_$(lpad(iter,5,"0")).txt"), vp*scale)

    open(joinpath(result_dir, "loss.txt"), "a") do io 
      writedlm(io, loss)
    end
  else
  end
end

if gpu
  LBFGS!(sess, loss, g, get_collection()[1]; callback=callback)
else
  LBFGS!(sess, loss, vars=[get_collection()[1]], callback=callback)
end
