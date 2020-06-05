using Revise
using ADSeismic
using ADCME
using MAT
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
# reset_default_graph()

# ap_sim = load_acoustic_model("models/marmousi2-model-true.mat")
# src = load_acoustic_source("models/marmousi2-model-true.mat")
# rcv = load_acoustic_receiver("models/marmousi2-model-true.mat")
# ## For debug, only run the first source
# # src = [src[1]]
# # rcv = [rcv[1]]

# if gpu
#   Rs_ = compute_forward_GPU(ap_sim, src, rcv)
# else
#   [SimulatedObservation!(ap_sim(src[i]), rcv[i]) for i = 1:length(src)]
#   Rs_ = [rcv[i].rcvv for i = 1:length(rcv)]
# end

# sess = Session(); init(sess)

# Rs = run(sess, Rs_)

# for i = 1:length(src)
#     writedlm(joinpath(output_dir, "marmousi-r$i.txt"), Rs[i])
# end

## visualize_wavefield if needed
# u = run(sess, ap_sim(src[div(length(src),2)+1]).u)
# visualize_wavefield(u, ap_sim.param)

################### Inversion using Automatic Differentiation #####################
reset_default_graph()

params = load_params("models/marmousi2-model-smooth.mat")
src = load_acoustic_source("models/marmousi2-model-smooth.mat")
rcv = load_acoustic_receiver("models/marmousi2-model-smooth.mat")
## For debug, only run the first source
# src = [src[1]]
# rcv = [rcv[1]]
vp0 = matread("models/marmousi2-model-smooth.mat")["vp"]
mean_vp0 = mean(vp0)
std_vp0 = std(vp0)
# vp_ = Variable(vp0 / mean_vp0)
# vp = vp_ * mean_vp0

## add NN
# nx = params.NX
# ny = params.NY
# dx = params.DELTAX
# dy = params.DELTAY
# z = zeros((nx+2)*(ny+2),2)
# for i = 1:nx+2
#  for j = 1:ny+2
#     z[(i-1)*(ny+2)+j, :] = [(i-1)*dx (j-1)*dy]
#   end
# end 
# vp = squeeze(fc(z, [10,1])) * mean_vp0
# vp = reshape(vp, size(vp0))
z = constant(rand(Float32, 1,8))
x = Generator(z, ratio=size(vp0)[1]/size(vp0)[2], vmin = -2, vmax = 2)
x = tf.cast(x, tf.float64)
x = x * std_vp0
vp = constant(vp0[:,:])
# x = tf.slice(x, (size(x).-size(vp0)).÷2, size(vp0))
# vp = vp + x
i = (size(vp0)[1]-size(x)[1])÷2 +1 :(size(vp0)[1]-size(x)[1])÷2 + size(x)[1]
j = (size(vp0)[2]-size(x)[2])÷2 +1 : (size(vp0)[2]-size(x)[2])÷2 + size(x)[2]
vp = scatter_add(vp, i, j, x)

# error()

## assemble acoustic propagator model
model = x->AcousticPropagatorSolver(params, x, vp^2)
vars = get_collection()
# vars = vp

## load data
Rs = Array{Array{Float64,2}}(undef, length(src))
for i = 1:length(src)
    Rs[i] = readdlm(joinpath(output_dir, "marmousi-r$i.txt"))
end

## optimization
if gpu
  # losses, grads = compute_loss_and_grads_GPU(model, src, rcv, Rs, get_collection())
  losses, grads = compute_loss_and_grads_GPU(model, src, rcv, Rs, vars)
  grad = sum(grads)
  loss = sum(losses)
else
  [SimulatedObservation!(model(src[i]), rcv[i]) for i = 1:length(src)]
  loss = sum([sum((rcv[i].rcvv-Rs[i])^2) for i = 1:length(rcv)]) 
end

sess = Session(); init(sess)
@info "Initial loss: ", run(sess, losses)

function callback(vs, iter, loss)
  if iter%10==0
    # if gpu
    #   x = Array(reshape(vs[1:(params.NY+2)*(params.NX+2)], params.NY+2, params.NX+2))
    # else
    #   x = vs[1]'
    # end
    x = run(sess, vp)'
    clf()
    pcolormesh([0:params.NX+1;]*params.DELTAX/1e3,[0:params.NY+1;]*params.DELTAY/1e3,  x)
    axis("scaled")
    colorbar(shrink=0.4)
    xlabel("x (km)")
    ylabel("z (km)")
    gca().invert_yaxis()
    title("Iteration = $iter")
    savefig(joinpath(figure_dir, "inv_$(lpad(iter,5,"0")).png"), bbox_inches="tight")
    writedlm(joinpath(result_dir, "inv_$(lpad(iter,5,"0")).txt"), x)

    open(joinpath(result_dir, "loss.txt"), "a") do io 
      writedlm(io, loss)
    end
  else
  end
end


if gpu
  LBFGS!(sess, loss, grad, vars; callback=callback)
  # LBFGS!(sess, loss, grad, vp_; callback=callback)
else
  LBFGS!(sess, loss, vars=[vars], callback=callback)
end


opt = AdamOptimizer(0.001).minimize(loss)
init(sess)
for i = 1:10000
  _, l_ = run(sess, [opt, loss])
  @info i, l_ 
end 