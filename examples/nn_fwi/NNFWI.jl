# ENV["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
using Revise
using ADSeismic
using ADCME
using MAT
using PyPlot
using Random
using DelimitedFiles
using Optim
using Dates
matplotlib.use("Agg")
close("all")

gpu = has_gpu() ? true : false

data_dir = "data/acoustic"
method = "NNFWI"

figure_dir = string("figure/",method,"/marmousi/")
result_dir = string("result/",method,"/marmousi/")
model_dir = string("NN_model/",method,"/marmousi/")
loss_file = joinpath(result_dir, "loss_$(Dates.now()).txt")

check_path(dir) = !ispath(data_dir) ? mkpath(dir) : nothing
check_path(figure_dir)
check_path(result_dir)
check_path(model_dir)

################### Inversion using Automatic Differentiation #####################
reset_default_graph()
model_name = "models/marmousi2-model-smooth-large.mat"
# model_name = "models/BP-model-smooth.mat"

## load model setting
params = load_params(model_name, vp_ref=3e3, PropagatorKernel=1)
src = load_acoustic_source(model_name)
rcv = load_acoustic_receiver(model_name)
vp0 = matread(model_name)["vp"]
mask = matread(model_name)["mask"]

## original fwi
# vp = Variable(vp0)

## fwi with fc
# nx, ny = params.NX, params.NY
# dx, dy = params.DELTAX, params.DELTAY
# z = zeros((nx+2)*(ny+2),2)
# for i = 1:nx+2
#  for j = 1:ny+2
#     z[(i-1)*(ny+2)+j, :] = [(i-1)*dx (j-1)*dy]
#   end
# end 
# init_guess = fc_init([2, 20, 20, 20, 1])
# θ = Variable(init_guess)
# c = fc(z, [20, 20, 20, 1], θ) * 100.0
# c = reshape(c, (nx+2, ny+2))
# vp = vp0 + c

## fwi with cnn
Random.seed!(123);
z = constant(rand(Float32, 1, 8))
x = Generator(z, ratio=size(vp0)[1]/size(vp0)[2], base=8)
x = tf.cast(x, tf.float64)
x = x * 1500
vp = add_initial_model(x, constant(vp0))
vp = mask .* vp + (1.0.-mask) .* vp0

## assemble acoustic propagator model
model = x->AcousticPropagatorSolver(params, x, vp)
vars = get_collection()

## load data
std_noise = 0
Random.seed!(1234);
Rs = Array{Array{Float64,2}}(undef, length(src))
for i = 1:length(src)
  Rs[i] = readdlm(joinpath(data_dir, "marmousi-r$i.txt"))
    # Rs[i] = readdlm(joinpath(data_dir, "BP-r$i.txt"))
  Rs[i] .+= randn(size(Rs[i])) .* std(Rs[i]) .* std_noise
end

## calculate loss and gradient
if gpu
  losses, grads = compute_loss_and_grads_GPU(model, src, rcv, Rs, vars)
  grad = sum(grads)
  loss = sum(losses)
else
  [SimulatedObservation!(model(src[i]), rcv[i]) for i = 1:length(src)]
  loss = sum([sum((rcv[i].rcvv-Rs[i])^2) for i = 1:length(rcv)])
  grad = gradients(loss)
end

## optimization
global_step = tf.Variable(0, trainable=false)
max_iter = 50000
# lr_decayed = tf.train.cosine_decay(0.001, global_step, max_iter)
lr_decayed = placeholder(0.0)
opt = AdamOptimizer(learning_rate=lr_decayed).minimize(loss, global_step=global_step, colocate_gradients_with_ops=true)

## run inversion
sess = Session(); init(sess)
loss0 = run(sess, loss)
@info "Initial loss: ", loss0

fp = open(loss_file, "w")
write(fp, "0,$loss0\n")
function callback(vs, iter, loss)
  if iter%50==1
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
  end
  if iter%1000 == 1
    ADCME.save(sess, joinpath(model_dir, "NNFWI_$(lpad(iter,5,"0")).mat"))
  end
  write(fp, "$iter,$loss\n")
  flush(fp)
end

## optimization using Adam
time = 0
for iter = 1:max_iter
  # global time += @elapsed  _, ls, lr = run(sess, [opt, loss, lr_decayed])
  if iter < 10000
    lr = iter/10000 * 1e-3
  else
    lr = (max_iter - iter)/(max_iter - 10000) * 1e-3
  end
  t = @elapsed  _, ls = run(sess, [opt, loss], feed_dict=Dict(lr_decayed=>lr))
  callback(nothing, iter, ls)
  println("$iter\t$ls\t$t")
end

## optimization using BFGS
Optimize!(sess, loss, vars=vars, grads=grad, callback=callback)

close(fp)


