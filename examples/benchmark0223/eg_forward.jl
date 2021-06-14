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

layers = ones(param.NX+2, param.NY+2)
n_piece = div(param.NX + 1, 3) + 1
for k = 1:3
    i_interval = (k-1)*n_piece+1:min(k*n_piece, param.NX+2)
    layers[:, i_interval] .= 0.5 + (k-1)*0.25
end

C = placeholder(3300*layers)
model = AcousticPropagatorSolver(param, src, C)

sess = Session(); init(sess)
u = run(sess, model.u)

p = visualize_wavefield(u, param)
saveanim(p, "wavefield.gif")

visualize_model(3300*layers, param)
savefig("model.png")


close("all")
plot(LinRange(0, 1, 1000), run(sess, rc))
xlabel("Time")
savefig("ricker.png")

@save "data.jld2" u 