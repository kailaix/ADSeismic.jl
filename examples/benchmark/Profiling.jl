using Revise
using ADCME
using ADSeismic
using PyPlot
using DelimitedFiles

##=======================
reset_default_graph()

scale = 100
param = AcousticPropagatorParams(NX=scale, NY=scale, NSTEP=3, DELTAT=1e-4,  DELTAX=1.0, DELTAY=1.0)

rc = Gauss(param, 1/pi, 100, 1e6)
srci = [div(param.NX,2)]
srcj = [div(param.NY,2)]
srcv = reshape(rc, :, 1)
src = AcousticSource(srci, srcj, srcv)

C = 3300*ones(param.NX+2, param.NY+2)
model = AcousticPropagatorSolver(param, src, C^2)

sess = Session(); init(sess)

run_profile(sess, model.u)
save_profile("acoustic-profile-$(has_gpu() ? "gpu" : "cpu").json")


##=======================
reset_default_graph()

scale = 100
param = ElasticPropagatorParams(NX=scale, NY=scale, NSTEP=2, DELTAT=1e-4,  DELTAX=1.0, DELTAY=1.0) 

source = Gauss(param, 1/pi, 100, 1e6)
srci = [div(param.NX,2)]
srcj = [div(param.NY,2)]
srctype = [0]
srcv = reshape(source, :, 1)
src = ElasticSource(srci, srcj, srctype, srcv)

vp = 3300.
vs = 3300. / 1.732
rho = 2800.
λ, ρ, μ = compute_default_properties(param.NX, param.NY, vp, vs, rho)
model = ElasticPropagatorSolver(param, src, ρ, λ, μ)

sess = Session(); init(sess)

run_profile(sess, model.vx)
save_profile("elastic-profile-$(has_gpu() ? "gpu" : "cpu").json")








