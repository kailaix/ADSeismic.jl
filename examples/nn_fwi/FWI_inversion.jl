ENV["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
using Revise
using ADSeismic
using ADCME
using PyPlot
using MAT
using Optim
using Random
using DelimitedFiles
matplotlib.use("Agg")
close("all")
if has_gpu()
  gpu = true
else
  gpu = false
end

data_dir = "data/acoustic"
if !ispath(data_dir)
  mkpath(data_dir)
end
figure_dir = "figure/FWI/marmousi/"
if !ispath(figure_dir)
  mkpath(figure_dir)
end
result_dir = "result/FWI/marmousi/"
if !ispath(result_dir)
  mkpath(result_dir)
end
if isfile(joinpath(result_dir, "loss.txt"))
  rm(joinpath(result_dir, "loss.txt"))
end


################### Inversion using Automatic Differentiation #####################
model_name = "models/marmousi2-model-smooth.mat"
# model_name = "models/BP-model-smooth.mat"

## load model setting
params = load_params(model_name)
src = load_acoustic_source(model_name)
rcv = load_acoustic_receiver(model_name)
vp = Variable(matread(model_name)["vp"])

## assemble acoustic propagator model
model = x->AcousticPropagatorSolver(params, x, vp^2)

## load data
std_noise = 0
Random.seed!(1234);
Rs = Array{Array{Float64,2}}(undef, length(src))
for i = 1:length(src)
    Rs[i] = readdlm(joinpath(data_dir, "marmousi-r$i.txt"))
    # Rs[i] = readdlm(joinpath(data_dir, "BP-r$i.txt"))
    Rs[i] .+= ( randn(size(Rs[i])) .+ mean(Rs[i]) ) .* std(Rs[i]) .* std_noise
end

## calculate loss
if gpu
  losses, grads = compute_loss_and_grads_GPU(model, src, rcv, Rs, vp)
  grad = sum(grads)
  loss = sum(losses)
else
  [SimulatedObservation!(model(src[i]), rcv[i]) for i = 1:length(src)]
  loss = sum([sum((rcv[i].rcvv-Rs[i])^2) for i = 1:length(rcv)]) 
end

global_step = tf.Variable(0, trainable=false)
max_iter = 50000
lr_decayed = tf.train.cosine_decay(1.0, global_step, max_iter)
opt = AdamOptimizer(lr_decayed).minimize(loss, global_step=global_step, colocate_gradients_with_ops=true)

sess = Session(); init(sess)
@info "Initial loss: ", run(sess, loss)

## run inversion
function callback(vs, iter, loss)
  if iter%10==0
    if gpu
      x = Array(reshape(vs[1:(params.NY+2)*(params.NX+2)], params.NY+2, params.NX+2))
    else
      x = vs[1]'
    end
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
  end
end


## BFGS
Optimize!(sess, loss, 10000, vars=[vp], grads=grad,  callback=callback)

## ADAM
# time = 0
# for iter = 0:max_iter
#   global time += @elapsed  _, ls, lr = run(sess, [opt, loss, lr_decayed])
#   callback(run(sess, vp)', iter, ls)
#   println("   $iter\t$ls\t$lr")
#   println(" * time: $time")
# end

