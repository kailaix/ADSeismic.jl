using ADSeismic
using ADCMEKit
using ADCME
using PyPlot
using DelimitedFiles
using JLD2
r = mpi_init()

M = N = Int(round(sqrt(mpi_size())))
n = 200÷M
@info M 
param = MPIAcousticPropagatorParams(NX = n*M, NY = n*N, n = n,
     Rcoef=0.2, DELTAX=20, DELTAY=20, DELTAT=0.05) 
compute_PML_Params!(param)

srci = [n*param.M÷5]
srcj = [n*param.N÷2]
srcv = hcat([Ricker(param, 100.0, 500.0)]...)
src = MPIAcousticSource(param, srci,srcj,srcv)

rcvj = Array(20:n*param.M-19)
rcvi = ones(Int64, length(rcvj))*((n*param.N)÷5)
receiver =  MPIAcousticReceiver(param, rcvi, rcvj)

c = 1000*ones(n, n)
center = n*M÷2
width = n*M ÷ 8
for i = 1:n 
    for j = 1:n 
        ii = i + (param.II-1) * n
        jj = j + (param.JJ-1) * n
        x = i*param.DELTAX + (param.II-1) * n * param.DELTAX
        y = j*param.DELTAY + (param.JJ-1) * n * param.DELTAY 
        if center -width <= ii <= center + width && center -width <=jj<=center +width
            c[i, j] = 2000.
        end
    end
end

propagator = MPIAcousticPropagatorSolver(param, src, c)
MPISimulatedObservation!(propagator, receiver)

sess = Session(); init(sess)

U = run(sess, propagator.u)
rcvv = run(sess, receiver.rcvv)

if mpi_size()==1
    close("all")
    pcolormesh(rcvv, cmap="gray")
    colorbar()
    xlabel("Location")
    ylabel("Time")
    savefig("receiver.png")

    close("all")
    visualize_wavefield(U, param)

    close("all")
    visualize_model(c, param)
    savefig("model.png")
end

@info "Saving..."
@save "data/dat$(mpi_rank())-$(mpi_size()).jld2" U rcvv


mpi_finalize()