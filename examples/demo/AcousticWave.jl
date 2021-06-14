using Revise
using ADCME
using ADSeismic
using PyPlot
using DelimitedFiles
matplotlib.use("agg")

param = AcousticPropagatorParams(NX=150, NY=150, NSTEP=1000, DELTAT=1e-4,  DELTAX=1.0, DELTAY=1.0, 
                                 vp_ref = 3000.0, Rcoef=0.001, 
                                 USE_PML_XMIN=true, USE_PML_XMAX=true, USE_PML_YMIN=true, USE_PML_YMAX=true)

rc = Ricker(param, 15.0, 100.0, 1e10) 
srci = [param.NX ÷ 2]
srcj = [param.NY ÷ 2]
srcv = reshape(rc, :, 1)
src = AcousticSource(srci, srcj, srcv)

c = 3000.0*ones(param.NX+2, param.NY+2)
model = AcousticPropagatorSolver(param, src, c)

sess = Session(); init(sess)
u = run(sess, model.u)

p = visualize_wavefield(u, param)
saveanim(p, "acoustic-wavefield.gif")
