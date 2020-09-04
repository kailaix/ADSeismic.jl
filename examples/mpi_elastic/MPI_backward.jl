using Revise
using ADSeismic
using ADCME 
using JLD2
using PyPlot
using ADCMEKit
ADCME.options.customop.verbose = false
mpi_init()
M = N = Int(sqrt(mpi_size()))
n = 10÷M
param = MPIElasticPropagatorParams(NX=10, NY=10, n=n, NSTEP=50, DELTAT=1e-4, 
    DELTAX=1.0, DELTAY=1.0,
    USE_PML_XMIN=false, USE_PML_XMAX = false, USE_PML_YMIN = false, USE_PML_YMAX = false)
compute_PML_Params!(param)
vp = 3300.
vs = 3300. / 1.732
rho = 2800.
_, ρ, μ = compute_default_properties(param, vp, vs, rho)
source = Ricker(param, 50.0, 200.0) 
source = ones(length(source))

pl = placeholder(zeros(1))
o = mpi_bcast(pl[1])

λ_normed = placeholder(ones(n+4, n+4)) 
λ_normed_ = mpi_bcast(λ_normed)
λ = 1e10 * λ_normed_

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
Vx, Vy = run(sess, [propagator.vx, propagator.vy], λ_normed => λ0)

loss = mpi_sum(sum((propagator.vx)^2) + sum((propagator.vy)^2))
g = gradients(loss, λ)
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


close("all")
gradview(sess, pl, loss, zeros(1), mpi=true)
mpi_rank()==0 && savefig("gradtest.png")

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