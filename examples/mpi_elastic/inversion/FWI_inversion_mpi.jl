using Revise
using ADSeismic
using ADCME
using PyPlot
using MAT
using DelimitedFiles
using JLD2
using PyCall
using ADOPT
using Dates
matplotlib.use("Agg")
close("all")

ADCME.options.customop.verbose = false

data_dir = "data/elastic"
method = "FWI/elastic"

figure_dir = string("figure/",method)
result_dir = string("result/",method)
loss_file = joinpath(result_dir, "loss_$(Dates.now()).txt")

check_path(dir) = !ispath(dir) ? mkpath(dir) : nothing
check_path(figure_dir)
check_path(result_dir)

################### Generate synthetic data #####################
reset_default_graph()
mpi_init()

# model_name = "models/marmousi2-model-true-large.mat"
# model_name = "models/marmousi2-model-true.mat"
# model_name = "models/BP-model-true.mat"
model_name = "models/marmousi2-model-smooth.mat"

param = load_params(model_name, "MPIElastic")
compute_PML_Params!(param)

vp0 = matread(model_name)["vp"]
vs0 = matread(model_name)["vs"]
# rho0 = matread(model_name)["rho"]
##
# vp0 = matread("models/marmousi2-model-true.mat")["vp"]
# vs0 = matread("models/marmousi2-model-true.mat")["vs"]
rho0 = matread("models/marmousi2-model-true.mat")["rho"]

@info "loading data..."
@load joinpath(data_dir, "data$(mpi_rank())-$(mpi_size()).jld2") Rs

vp_ = Variable(vp0)
# vs_ = Variable(vs0)
ps_ratio_ = Variable(1.75)
# vs_ = vp0 / ps_ratio_
# rho_ = Variable(rho0)
var_size = size(vp_)
if exists(matopen(model_name), "mask")
    @info "using mask..."
    mask = matread(model_name)["mask"] 
    vp0 = mask .* vp_ + (1.0.-mask) .* vp0
    # vs0 = mask .* vs_ + (1.0.-mask) .* vs0
    # rho0 = mask .* rho_ + (1.0.-mask) .* rho0
else
    vp0 = vp_
    vs0 = vs_
end

vp = mpi_bcast(vp0)
ps_ratio = mpi_bcast(ps_ratio_, deps=sum(vp))
# vs = vs0
ps_ratio = tanh(ps_ratio - 1.75)*0.5 + 1.75
vs = vp / ps_ratio
# vs = mpi_bcast(vs0, deps=sum(vp))
# vs = constant(vs0)
rho = constant(rho0)

vp = extract_local_patch(param, vp, tag = 0)
vs = extract_local_patch(param, vs, deps = sum(vp), tag = 810000)
rho = extract_local_patch(param, rho, deps = sum(vs), tag = 820000)

λ, μ, ρ = compute_lame_parameters(vp, vs, rho)

src = load_elastic_source(model_name, use_mpi=true, param=param)
receiver = load_elastic_receiver(model_name, use_mpi=true, param=param)

Rs_ = []
Vs_ = []

local_loss = constant(0.0)
for i = 1:length(src)
  
  rcvv = Rs[i]
  src_ = src[i]
  receiver_ = receiver[i]

  propagator = MPIElasticPropagatorSolver(param, src_, ρ, λ, μ; tag_offset = i*100000, dep = length(Vs_)==0 ? nothing : sum(Vs_[end]))
  MPISimulatedObservation!(propagator, receiver_)
  
  push!(Rs_, receiver_.rcvv)
  push!(Vs_, propagator.vy)
  
  if !isnothing(rcvv)
      global local_loss += sum((receiver_.rcvv - rcvv)^2) 
  end

end

local_loss += sum(Vs_[end]) * 1e-20

loss = mpi_sum(local_loss) * 1e7
# g = gradients(loss, [vp_, vs_])
g = gradients(loss, [vp_, ps_ratio_])
@info g
g1 = reshape(g[1], [prod(var_size)])
g2 = reshape(g[2], [1])
g = concat([g1,g2], dims=1)
# g = tf.concat(g, axis=1)
# g = concat(g, dims=2)

py"""
from tensorflow.core.protobuf import rewriter_config_pb2
import tensorflow as tf
config = tf.ConfigProto()
off = rewriter_config_pb2.RewriterConfig.OFF
config.graph_options.rewrite_options.memory_optimization  = off
"""
session_conf = py"config"
sess = Session(config = session_conf); init(sess)
#sess = Session(); init(sess)

# @info "running loss"
# @time x = run(sess, loss)
# @info "running gradients"
# @time x = run(sess, g)

#using ADOPT
@info "starting..."
@info "initial local loss = $(run(sess, local_loss)) at rank = $(mpi_rank())"

function calculate_loss(x)
    # x = reshape(x, var_size)
    p = reshape(x[1:prod(var_size)], var_size)
    # s = reshape(x[prod(var_size)+1:2*prod(var_size)], var_size)
    ps =  x[prod(var_size)+1]
    # r = reshape(x[2*prod(var_size)+1:3*prod(var_size)], var_size)
    # L = run(sess, loss, vp_=>p, vs_=>s)#, rho_=>r)
    L = run(sess, loss, vp_=>p, ps_ratio_=>ps)
    # L = run(sess, loss, vp_=>p)
    L
end

iter = 0
iter_result = Array{Float64, 2}[]
function calculate_gradients(G, x)
    # x = reshape(x, var_size)
    p = reshape(x[1:prod(var_size)], var_size)
    # s = reshape(x[prod(var_size)+1:2*prod(var_size)], var_size)
    ps =  x[prod(var_size)+1]
    s = p ./ ps
    # r = reshape(x[2*prod(var_size)+1:3*prod(var_size)], var_size)
    # G[:] = run(sess, g, vp_=>p, vs_=>s)#, rho_=>r)
    G[:] = run(sess, g, vp_=>p, ps_ratio_=>ps)
    # L = run(sess, loss, vp_=>p)
    if mpi_rank() == 0
      @info "iter = ", iter
      if iter % 1 == 0
          clf()
          subplot(211)
          pcolormesh(p', shading="auto", cmap="jet")
          axis("scaled")
          colorbar()
          xlabel("x (km)")
          ylabel("z (km)")
          gca().invert_yaxis()
          title("Iteration = $iter")
        #   savefig(joinpath(figure_dir, "inv_p_$(lpad(iter,5,"0")).png"), bbox_inches="tight")
        #   clf()
          subplot(212)
          pcolormesh(s', shading="auto", cmap="jet")
          axis("scaled")
          colorbar()
          xlabel("x (km)")
          ylabel("z (km)")
          gca().invert_yaxis()
        #   subplot(312)
        #   pcolormesh(r', cmap="jet")
        #   gca().invert_yaxis()
        #   colorbar()
          savefig(joinpath(figure_dir, "inv_$(lpad(iter,5,"0")).png"), bbox_inches="tight")
          writedlm(joinpath(result_dir, "inv_vp_$(lpad(iter,5,"0")).txt"), p)
          writedlm(joinpath(result_dir, "inv_vs_$(lpad(iter,5,"0")).txt"), s)
      end
      global iter += 1
    end
end

losses = Float64[]
function step_callback(x)
    @info "loss = $x"
    push!(losses, x)
end

# initial_x = run(sess, [vp_, vs_])#, rho_])
initial_x = run(sess, [vp_, ps_ratio_])
# initial_x = cat(initial_x..., dims=2)[:]
initial_x = append!(reshape(initial_x[1], prod(var_size)), initial_x[2])
options = Options()
result = ADOPT.mpi_optimize(calculate_loss, calculate_gradients, initial_x, LBFGS(), options; step_callback = step_callback)

if mpi_size()>1
    mpi_finalize()  
end
