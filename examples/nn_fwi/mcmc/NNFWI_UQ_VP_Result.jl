ENV["CUDA_VISIBLE_DEVICES"] = "2,3"
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
use_gpu(0)

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
model_name = "../models/marmousi2-model-smooth.mat"

## load model setting
params = load_params(model_name)

results = readdlm("Data/UQ_VP.txt")
@show size(results)
M = reshape(mean(results[1:end,:], dims = 1), params.NX+2, params.NY+2)
V = reshape(std(results[1:end,:], dims = 1), params.NX+2, params.NY+2)

close("all")
visualize_model(M, params)
savefig("VP_mean.png")

close("all")
visualize_model(V, params)
savefig("VP_std.png")