# using Revise
using ADSeismic
using ADCME
using PyPlot
using MAT
using DelimitedFiles
using JLD2
using PyCall
matplotlib.use("Agg")
close("all")

ADCME.options.customop.verbose = false

#################################################################
output_dir = "data/elastic"
check_path(dir) = !ispath(dir) ? mkpath(dir) : nothing
check_path(output_dir)

################### Generate synthetic data #####################
reset_default_graph()
mpi_init()

# model_name = "models/marmousi2-model-true-large.mat"
model_name = "models/marmousi2-model-true.mat"
# model_name = "models/BP-model-true.mat"

param = load_params(model_name, "MPIElastic")
@info param.NSTEP, param.n
compute_PML_Params!(param)

vp = matread(model_name)["vp"]
vs = matread(model_name)["vs"]
rho = matread(model_name)["rho"]

vp = extract_local_patch(param, vp, tag = 0)
vs = extract_local_patch(param, vs, deps = sum(vp), tag = 810000)
rho = extract_local_patch(param, rho, deps = sum(vs), tag = 820000)

λ, μ, ρ = compute_lame_parameters(vp, vs, rho)

src = load_elastic_source(model_name, use_mpi=true, param=param)
receiver = load_elastic_receiver(model_name, use_mpi=true, param=param)

Rs_ = []
Vs_ = []
for i = 1:length(src)
  src_ = src[i]
  receiver_ = receiver[i]

  propagator = MPIElasticPropagatorSolver(param, src_, ρ, λ, μ; tag_offset = i*100000, dep = length(Vs_)==0 ? nothing : sum(Vs_[end]))
  MPISimulatedObservation!(propagator, receiver_)

  push!(Rs_, receiver_.rcvv)
  push!(Vs_, propagator.vy)
end

sess = Session(); init(sess)

@info "starting..."

@time Rs = run(sess, Rs_)
@info "saving..."
@save joinpath(output_dir, "data$(mpi_rank())-$(mpi_size()).jld2") Rs

if mpi_rank()==0 && mpi_size()==1
  ii = 1 # source number
  u = run(sess, propagator(src[ii]).vx)
  p = visualize_wavefield(u, param)
  saveanim(p, "marmousi-wavefield.gif")
end

if mpi_size()>1
    mpi_finalize()  
end
