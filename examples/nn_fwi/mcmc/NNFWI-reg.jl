include("readargs.jl")
using Revise
using ADSeismic
using ADCME
using MAT
using PyPlot
using DelimitedFiles
using LineSearches
using Mamba
using ProgressMeter
using JLD2
using Statistics 
using Random
# matplotlib.use("Agg")
close("all")
if has_gpu()
  gpu = true
else
  gpu = false
end


# creating data directories 
data_dir = "../data/acoustic"
if !ispath(data_dir)
  mkpath(data_dir)
end
figure_dir = "../figure/acoustic/reg3/"
if !ispath(figure_dir)
  mkpath(figure_dir)
end
result_dir = "../result/acoustic/reg3/"
if !ispath(result_dir)
  mkpath(result_dir)
end

################### Inversion using Automatic Differentiation #####################
reset_default_graph()
model_name = "../models/marmousi2-model-smooth.mat"

## load model setting
params = load_params(model_name)
src = load_acoustic_source(model_name)
rcv = load_acoustic_receiver(model_name)
vp0 = matread(model_name)["vp"]
mean_vp0 = mean(vp0)
std_vp0 = std(vp0)

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
std_noise = 0
Random.seed!(1234);
Rs = Array{Array{Float64,2}}(undef, length(src))
for i = 1:length(src)
    Rs[i] = readdlm(joinpath(data_dir, "marmousi-r$i.txt"))
    Rs[i] .+= randn(size(Rs[i])) .* std(Rs[i]) .* std_noise
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


vp_transpose = vp' 

sess = Session(allow_growth=true); init(sess)
ADCME.load(sess, "NNFWI-UQ.mat")
@info "Initial loss: ", run(sess, loss)

w0 = pack(run(sess, get_collection()))


σ = 1.0
σx = 1000000.0
function logf(x)
    ws = unpack(x, get_collection())
    d = Dict(
      [x=>y for (x,y) in zip(get_collection(), ws)]...
    )
    loss_value = run(sess, loss, feed_dict=d)
    -loss_value/2σ^2 - sum(x.^2)/2σx^2
end

n = nsim
burnin = nsim÷5
sim = Chains(n, length(w0))

θ = RWMVariate(w0, scaling*ones(length(w0)), logf, proposal = SymUniform)


@showprogress for i = 1:n
  sample!(θ)
  sim[i,:,1] = θ
end


v = sim.value
K = zeros(size(vp_transpose)..., n-burnin)
STD = zeros(size(vp_transpose)...)
MEAN = zeros(size(vp_transpose)...)

@showprogress for i = 1:n-burnin
  ws = v[i+burnin,:,1]
  d = Dict(
      [x=>y for (x,y) in zip(get_collection(), ws)]...
    )
  K[:,:,i] = run(sess, vp_transpose, feed_dict = d)
end

for i = 1:size(STD,1)
  for j = 1:size(STD, 2)
    STD[i,j] = std(K[i,j,:])
    MEAN[i,j] = mean(K[i,j,:])
  end
end

if !isdir("Data/$label")
  mkpath("Data/$label")
end

@save "Data/$label/dat.jld2" v K STD MEAN

x = run(sess, STD)'
clf()
pcolormesh([0:params.NX+1;]*params.DELTAX/1e3,[0:params.NY+1;]*params.DELTAY/1e3,  x)
axis("scaled")
colorbar(shrink=0.4)
xlabel("x (km)")
ylabel("z (km)")
gca().invert_yaxis()
title("Iteration = $iter")
savefig("Data/$label/std.png", bbox_inches="tight")
