using Revise
using ADSeismic
using ADCME
using PyPlot
using MAT
using DelimitedFiles
using JLD2
using ADOPT
using PyCall
matplotlib.use("Agg")
close("all")

ADCME.options.customop.verbose = false

py"""
import tensorflow as tf
from tensorflow.core.protobuf import rewriter_config_pb2
config_proto = tf.ConfigProto(intra_op_parallelism_threads=0,
                              inter_op_parallelism_threads=0)
# off = rewriter_config_pb2.RewriterConfig.OFF
# config_proto.graph_options.rewrite_options.memory_optimization  = off
"""
config_proto = py"config_proto"
@show config_proto

output_dir = "data/elastic"
if !ispath(output_dir)
  mkpath(output_dir)
end

################### Generate synthetic data #####################
reset_default_graph()
mpi_init()

# model_name = "models/marmousi2-model-true-large.mat"
# model_name = "models/marmousi2-model-true.mat"
# model_name = "models/BP-model-true.mat"
model_name = "models/marmousi2-model-smooth.mat"

param = load_params(model_name, "MPIElastic")
compute_PML_Params!(param)

vp = matread(model_name)["vp"]
vs = matread(model_name)["vs"]
rho = matread(model_name)["rho"]
##
vs = matread("models/marmousi2-model-true.mat")["vs"]
rho = matread("models/marmousi2-model-true.mat")["rho"]

@info "loading data..."
@load "data/data$(mpi_rank())-$(mpi_size()).jld2" Rs

vp_ = Variable(vp)
var_size = size(vp_)

vp = mpi_bcast(vp_)
vs = constant(vs)
rho = constant(rho)

vp = extract_local_patch(param, vp)
vs = extract_local_patch(param, vs)
rho = extract_local_patch(param, rho)

λ, μ, ρ = compute_lame_parameters(vp, vs, rho)

# src = load_elastic_source(model_name, use_mpi=true, param=param)[4:4]
# receiver = load_elastic_receiver(model_name, use_mpi=true, param=param)[4:4]
# propagator = x->MPIElasticPropagatorSolver(param, x, ρ, λ, μ)
# [MPISimulatedObservation!(propagator(src[i]), receiver[i]) for i = 1:length(src)]

src = load_elastic_source(model_name, use_mpi=true, param=param)
receiver = load_elastic_receiver(model_name, use_mpi=true, param=param)

rcvv = Rs[1]
for i = 1:4

  src_ = src[i]
  receiver_ = receiver[i]

  propagator = MPIElasticPropagatorSolver(param, src_, ρ, λ, μ; dep = length(Rs_)==0 ? nothing : sum(Rs_[end]))
  MPISimulatedObservation!(propagator, receiver_)

  if !isnothing(rcvv)
      global local_loss += sum((receiver_.rcvv - rcvv )^2) 
  end

end

loss = mpi_sum(local_loss) * 1e7
g = gradients(loss, vp_)

sess = Session(config=config_proto); init(sess)

@info "running loss"
run(sess, loss)
@info "running gradients"
run(sess, g)

@info "starting..."
@info "initial local loss = $(run(sess, local_loss)) at rank = $(mpi_rank())"

function calculate_loss(x)
    x = reshape(x, var_size)
    L = run(sess, loss, vp_=>x)
    L
end

iter = 0
iter_result = Array{Float64, 2}[]
function calculate_gradients(G, x)
    x = reshape(x, var_size)
    G[:] = run(sess, g, vp_=>x)

    global iter += 1
    if iter % 100 == 1
      clf()
      pcolormesh(x', cmap="jet")
      savefig("test_figures/x-$(iter).png")
    end
end

losses = Float64[]
function step_callback(x)
    @info "loss = $x"
    push!(losses, x)
end

initial_x = run(sess, vp_)
options = Options()
result = ADOPT.mpi_optimize(calculate_loss, calculate_gradients, initial_x[:], LBFGS(), options; step_callback = step_callback)

if mpi_rank()==0 && mpi_size()==1
  ii = 1 # source number
  u = run(sess, propagator(src[ii]).vx)
  p = visualize_wavefield(u, param)
  saveanim(p, "marmousi-wavefield.gif")
end

if mpi_size()>1
    mpi_finalize()  
end
