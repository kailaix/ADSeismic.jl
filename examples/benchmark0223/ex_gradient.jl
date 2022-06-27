using ADCME
using ADSeismic
using PyPlot
using DelimitedFiles
using JLD2 

scale = 201
 
param = AcousticPropagatorParams(NX=scale, NY=scale, 
    NSTEP=1000, DELTAT=1e-4,  DELTAX=1.0, DELTAY=1.0,
    PropagatorKernel = 0, Rcoef = 1e-8)

rc = Ricker(param, 30.0, 200.0, 1e6)
srci = [div(param.NX,2)]
srcj = [div(param.NY,5)]
srcv = reshape(rc, :, 1)
src = AcousticSource(srci, srcj, srcv)


C = Variable(3300*ones(param.NX+2, param.NY+2))
model = AcousticPropagatorSolver(param, src, C)
@load "data.jld2" u 

U = model.u
loss = sum((u[:, :, 40] - U[:,:,40])^2)
g = gradients(loss, C)

sess = Session(); init(sess)
G = run(sess, g)

close("all")
G[:,1:49] .= NaN 
pcolormesh(G')
gca().invert_yaxis()
savefig("gradient.png")
