using Revise
using ADCME
using ADSeismic
using PyPlot
using DelimitedFiles
matplotlib.use("agg")

param = ElasticPropagatorParams(NX=150, NY=150, NSTEP=500, DELTAT=1e-4, DELTAX=1.0, DELTAY=1.0, 
                                vp_ref = 3300.0, USE_PML_XMIN=true, USE_PML_XMAX=true, USE_PML_YMIN=true, USE_PML_YMAX=true) 

source = Ricker(param, 15.0, 100.0, 1e6) 
srci = [param.NX ÷ 2]
srcj = [param.NY ÷ 2]
srctype = [0] #0: velocity; 1: stress
srcv = reshape(source, :, 1)

# srci = [param.NX ÷ 2, param.NX ÷ 2]
# srcj = [param.NY ÷ 2, param.NY ÷ 2]
# srctype = [2,3]
# srcv = tf.concat([reshape(source, :, 1), reshape(source, :, 1)],1)
src = ElasticSource(srci, srcj, srctype, srcv)

vp = 3000.
vs = vp / 1.732
rho = 2800.
λ, μ, ρ = compute_lame_parameters(param.NX, param.NY, vp, vs, rho)
model = ElasticPropagatorSolver(param, src, ρ, λ, μ)

sess = Session(); init(sess)    
vx = run(sess, model.vx)

p = visualize_wavefield(vx, param)
saveanim(p, "elastic-wavefield.gif")