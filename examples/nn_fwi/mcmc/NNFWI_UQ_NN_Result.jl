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
σv_ = 0.0
for i = 1:length(src)
    Rs[i] = readdlm(joinpath(data_dir, "marmousi-r$i.txt"))
    Rs[i] .+= randn(size(Rs[i])) .* std(Rs[i]) .* std_noise
    global σv_ += std(Rs[i])
end
σv_ /= length(src) ## data error std

Rs_ = compute_forward_GPU(model, src, rcv)
loss = sum([sum((x-y)^2) for (x,y) in zip(Rs, Rs_)])

sess = Session(); init(sess)
ADCME.load(sess, "NNFWI_05001.mat")
@info run(sess, loss)


results = zeros(4901, size(vp, 1), size(vp, 2))
nn = readdlm("Data/UQ_NN.txt")
for i = 1:4901
  @info i 
  r = unpack(nn[i,:], get_collection())
  results[i,:,:] = run(sess, vp, feed_dict = Dict(
    [x=>y for (x,y) in zip(get_collection(), r)]
  ))
end

M = mean(results[1001:end,:,:], dims = 1)[1,:,:]
V = std(results[1001:end,:,:], dims = 1)[1,:,:]

close("all")
visualize_model(M, params)
savefig("NN_mean.png")


close("all")
visualize_model(V, params)
savefig("NN_std.png")

close("all")
visualize_model(V ./ M , params)
savefig("NN_div.png")
