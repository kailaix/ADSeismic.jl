# mpirun -n 16 julia MPI_forward.jl
using Revise
using ADSeismic
using ADCME 
using JLD2
ADCME.options.customop.verbose = false
mpi_init()
M = N = Int(sqrt(mpi_size()))
n = 100÷M

param = MPIElasticPropagatorParams(NX=100, NY=100, n=n, NSTEP=1000, DELTAT=1e-4, 
    DELTAX=1.0, DELTAY=1.0,
    USE_PML_XMIN=true, USE_PML_XMAX = true, USE_PML_YMIN = true, USE_PML_YMAX = true)
compute_PML_Params!(param)
vp = 3300.
vs = 3300. / 1.732
rho = 2800.
λ, ρ, μ = compute_default_properties(param, vp, vs, rho)
source = Ricker(param, 50.0, 200.0) 

# close("all")
# plot(run(sess, source))
# savefig("test.png")
srci = [div(param.NX,5)]
srcj = [div(param.NY,2)]
srctype = [0]
srcv = reshape(source, :, 1)
src = MPIElasticSource(param, srci, srcj, srctype, srcv)

propagator = MPIElasticPropagatorSolver(param, src, ρ, λ, μ)

sess = Session(); init(sess)
@info "starting..."
Vx, Vy, Sxx, Syy, Sxy = run(sess, [propagator.vx, propagator.vy, propagator.sigmaxx, propagator.sigmayy, propagator.sigmaxy])

if mpi_size()>1
    mpi_finalize()  
end
if mpi_rank()==0 && mpi_size()==1
    # close("all")
    # pcolormesh(rcvv, cmap="gray")
    # colorbar()
    # xlabel("Location")
    # ylabel("Time")
    # savefig("receiver.png")

    close("all")
    visualize_wavefield(Vx, param)

end