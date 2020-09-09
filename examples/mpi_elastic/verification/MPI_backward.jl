using Revise
using ADSeismic
using ADCME 
using JLD2
using PyPlot
# using ADCMEKit
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
_, ρ, μ = compute_default_properties(param, vp, vs, rho)
source = Ricker(param, 50.0, 200.0) 
source = ones(length(source))

pl = placeholder(zeros(1))
o = mpi_bcast(pl[1])

pl = placeholder(ones(1))
λ_local = placeholder(ones(n, n)) + sum(pl)
λ_ext = mpi_halo_exchange2(λ_local, param.M, param.N)
λ = 1e10 * λ_ext

srci = [div(param.NX,5)]
srcj = [div(param.NY,2)]
srctype = [0]
srcv = reshape(source, :, 1)
src = MPIElasticSource(param, srci, srcj, srctype, srcv)

propagator = MPIElasticPropagatorSolver(param, src, ρ, λ, μ)

sess = Session(); init(sess)

λ0 = zeros(n+4, n+4)
for i = 1:n+4
    for j = 1:n+4
        x = (i-2) + (param.II-1)*param.n
        y = (j-2) + (param.JJ-1)*param.n
        if x>2div(param.NX, 5) && x<3div(param.NX,5) && y>2div(param.NY, 5) && y<3div(param.NY,5) 
            λ0[i,j] = 2.0
        else
            λ0[i,j] = 1.0
        end
    end
end
@info "starts..."
Vx, Vy = run(sess, [propagator.vx, propagator.vy], λ_local => λ0[3:end-2, 3:end-2])

loss = mpi_sum(sum((propagator.sigmaxx)^2) ) #+ sum((propagator.sigmaxx)^2) + sum((propagator.sigmayy)^2)  + sum((propagator.sigmaxy)^2))  
g = gradients(loss, λ_local)

C = run(sess, g)
VX = Vx[end,:,:]
if mpi_size()==1
    @save "data/C.jld2" C VX
else 
    @save "data/C$(mpi_rank()).jld2" C VX
end
# @info "Start running..."
# @info run(sess, loss)
# # @info "Finished..."
# # # @info "starting..."

# C = run(sess, g, λ_normed => ones(n+4, n+4))

# close("all")
# pcolormesh(C)
# colorbar()
# if mpi_size()==1
#     savefig("C.png")
# else
#     savefig("C$(mpi_rank()).png")
# end

# if mpi_size()==1
#     close("all")
#     pcolormesh(C)
#     colorbar()
#     savefig("gradient.png")
#     close("all")
#     visualize_wavefield(Vx, param)
# end


# close("all")
# gradview(sess, pl, loss, zeros(1), mpi=true)
# mpi_rank()==0 && savefig("gradtest.png")

if mpi_size()>1
    mpi_finalize()
end

# # Vx = run(sess, propagator.vx)

# # Vx[2,:,:]
# @info "saving..."
# @save "data/dat$(mpi_rank())-$(mpi_size()).jld2" Vx Vy Sxx Syy Sxy param

# if mpi_size()>1
#     mpi_finalize()  
# end
# if mpi_rank()==0 && mpi_size()==1
#     # close("all")
#     # pcolormesh(rcvv, cmap="gray")
#     # colorbar()
#     # xlabel("Location")
#     # ylabel("Time")
#     # savefig("receiver.png")

#     close("all")
#     visualize_wavefield(Vx, param)

# end