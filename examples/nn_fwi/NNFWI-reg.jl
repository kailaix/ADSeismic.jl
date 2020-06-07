using Revise
using ADSeismic
using ADCME
using MAT
using PyPlot
using DelimitedFiles
# matplotlib.use("Agg")
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
figure_dir = "figure/acoustic/reg/"
if !ispath(figure_dir)
  mkpath(figure_dir)
end
result_dir = "result/acoustic/reg/"
if !ispath(result_dir)
  mkpath(result_dir)
end
# if isfile(joinpath(result_dir, "loss.txt"))
#   rm(joinpath(result_dir, "loss.txt"))
# end

################### Inversion using Automatic Differentiation #####################
reset_default_graph()
model_name = "models/marmousi2-model-smooth.mat"

## load model setting
params = load_params(model_name)
src = load_acoustic_source(model_name)
rcv = load_acoustic_receiver(model_name)
vp0 = matread(model_name)["vp"]
mean_vp0 = mean(vp0)
std_vp0 = std(vp0)

## original fwi
# vp_ = Variable(vp0 / mean_vp0)
# vp = vp_ * mean_vp0
# vars = vp_

## fwi with NN
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
x = Generator(z, ratio=size(vp0)[1]/size(vp0)[2], base=4)
x = tf.cast(x, tf.float64)
x = x * std_vp0
vp = constant(vp0[:,:])
if size(x) <= size(vp)
  i = (size(vp0)[1]-size(x)[1])÷2 +1 :(size(vp0)[1]-size(x)[1])÷2 + size(x)[1]
  j = (size(vp0)[2]-size(x)[2])÷2 +1 : (size(vp0)[2]-size(x)[2])÷2 + size(x)[2]
  vp = scatter_add(vp, i, j, x)
elseif  size(x) > size(vp)
  vp = vp + tf.slice(x, (size(x).-size(vp0)).÷2, size(vp0))
else
  error("Size error: ", size(vp), size(x))
end

## assemble acoustic propagator model
model = x->AcousticPropagatorSolver(params, x, vp^2)
vars = get_collection()

## load data
Rs = Array{Array{Float64,2}}(undef, length(src))
for i = 1:length(src)
    Rs[i] = readdlm(joinpath(data_dir, "marmousi-r$i.txt"))
end

## calculate loss
if gpu
  losses, grads = compute_loss_and_grads_GPU(model, src, rcv, Rs, vars)
  grad = sum(grads)
  loss = sum(losses)
else
  [SimulatedObservation!(model(src[i]), rcv[i]) for i = 1:length(src)]
  loss = sum([sum((rcv[i].rcvv-Rs[i])^2) for i = 1:length(rcv)]) 
end

global_step = tf.Variable(0, trainable=false)
max_iter = 1000
lr_decayed = tf.train.cosine_decay(0.001, global_step, max_iter)
opt = AdamOptimizer(lr_decayed).minimize(loss, global_step=global_step, colocate_gradients_with_ops=true)

sess = Session(); init(sess)
@info "Initial loss: ", run(sess, loss)

## run inversion
function callback(vs, iter, loss)
  if iter%10==0
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
  end
end

## optimization using Adam
time = 0
for iter = 1:max_iter
  global time += @elapsed  _, ls, lr = run(sess, [opt, loss, lr_decayed])
  callback(nothing, iter, ls)
  println("   $iter\t$ls\t$lr")
  println(" * time: $time")
end 

## optimization using L-BFGS
if gpu
  LBFGS!(sess, loss, grad, vars; callback=callback)
else
  LBFGS!(sess, loss, vars=[vars], callback=callback)
end

