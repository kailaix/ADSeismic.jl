ENV["CUDA_VISIBLE_DEVICES"] = "6,7"
using Revise
using ADSeismic
using ADCME
using MAT
using PyPlot
using DelimitedFiles
using LineSearches
using Random
using Mamba
using ProgressMeter

# matplotlib.use("Agg")
close("all")

data_dir = "../data/acoustic"
if !ispath(data_dir)
  mkpath(data_dir)
end
figure_dir = "figure/"
if !ispath(figure_dir)
  mkpath(figure_dir)
end
result_dir = "result/"
if !ispath(result_dir)
  mkpath(result_dir)
end
# if isfile(joinpath(result_dir, "loss.txt"))
#   rm(joinpath(result_dir, "loss.txt"))
# end

################### Inversion using Automatic Differentiation #####################
reset_default_graph()
model_name = "../models/marmousi2-model-smooth.mat"
vp_true = matread("../models/marmousi2-model-true.mat")["vp"]

## load model setting
params = load_params(model_name)
src = load_acoustic_source(model_name)
rcv = load_acoustic_receiver(model_name)
vp0 = matread(model_name)["vp"]
mean_vp0 = mean(vp0)
std_vp0 = std(vp0)

# construct a neural, no dropout is present 
Random.seed!(0);
z = constant(rand(Float32, 1,8))
x = Generator(z, ratio=size(vp0)[1]/size(vp0)[2], base=4)
x = cast(Float64, x)
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
σv = 0.0
for i = 1:length(src)
    Rs[i] = readdlm(joinpath(data_dir, "marmousi-r$i.txt"))
    Rs[i] .+= randn(size(Rs[i])) .* std(Rs[i]) .* std_noise
    global σv += std(Rs[i])
end
σv /= length(src) ## data error std

Rs_ = compute_forward_GPU(model, src, rcv)
loss = sum([sum((x-y)^2) for (x,y) in zip(Rs, Rs_)])

sess = Session(); init(sess)
ADCME.load(sess, "NNFWI_05001.mat")
@info run(sess, loss)

vp_ = run(sess, vp)
visualize_model(vp_, params)
savefig("test-vp-nn.png")

σm = std(pack(run(sess, get_collection()))) ## data error std

## apply some scaling
σm *= 50
σv *= 50

function logf(x)
  vals = unpack(x, get_collection())
  loss_val = run(sess, loss, feed_dict = 
    Dict([a=>b for (a,b) in zip(get_collection(), vals)])
  )
  p1 = loss_val/(2σv^2)
  p2 = sum((x.-θ0).^2)/(2σm^2)
  -p1-p2
end

n = 1000000
burnin = 1000
sim = Chains(n, 1)
θ0 = pack(run(sess, get_collection()))
theta = RWMVariate(θ0, 0.03*ones(length(θ0)), logf, proposal = SymUniform)
results = zeros(n, length(θ0))

v = theta
num_accept = 0
accept = true
for i in 1:n
  t0 = time() 
  # sample!(theta)
  if accept
    global prob_v = logf(v.value)
  end
  x = v + v.tune.scale .* rand(v.tune.proposal(0.0, 1.0), length(v))
  prob_x = logf(x)
  if rand() < exp(prob_x - prob_v)
    v[:] = x
    global num_accept += 1
    global accept = true
  else
    global accept = false
  end
  results[i,:] = theta.value
  if mod(i, 100)==1
    writedlm("Data/UQ_NN.txt", results[1:i,:])
  end
  # @info i, "time consumed:", time() - t0
  @info i, "accept rate:", num_accept/i, "prob_v:", exp(prob_v)
  @info "exp(prob_x - prob_v)", exp(prob_x - prob_v)
end



