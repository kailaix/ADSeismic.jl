export MPIAcousticPropagatorParams, MPIAcousticSource, MPIAcousticReceiver, MPIAcousticPropagator, compute_PML_Params!,
        MPIAcousticPropagatorSolver, MPISimulatedObservation!, MPIAcousticReceiver
@with_kw mutable struct MPIAcousticPropagatorParams
    # number of grids along x,y axis and time steps
    NX::Int64 = 0
    NY::Int64 = 0 
    M::Int64 = 0
    N::Int64 = 0
    n::Int64 = 300
    NSTEP::Int64 = 2000 * 2 

    # size of grid cell and time step
    DELTAX::Float64 = 10.
    DELTAY::Float64 = 10.
    DELTAT::Float64 = 2.e-3 / 2

    # PML boundary conditon 
    USE_PML_XMIN::Bool = true
    USE_PML_XMAX::Bool = true
    USE_PML_YMIN::Bool = true
    USE_PML_YMAX::Bool = true
    NPOINTS_PML::Int64 = 12
    NPOWER::Int64 = 2
    damping_x::Union{Missing,Float64} = missing
    damping_y::Union{Missing,Float64} = missing
    Rcoef::Float64 = 0.001 # Relative reflection coefficient
    vp_ref::Float64 = 1000. 

    # Auxilliary Data
    Σx::Array{Float64} = []
    Σy::Array{Float64} = []
    IJ::Array{Int64} = []
    IJn::Array{Int64}  = []
    IJp::Array{Int64} = []
    IpJ::Array{Int64} = []
    IpJp::Array{Int64} = []
    IpJn::Array{Int64} = []
    InJ::Array{Int64} = []
    InJn::Array{Int64} = []
    InJp::Array{Int64} = []
    
    # display params
    IT_DISPLAY::Int64 = 0

    # mpi 
    II::Int64 = -1 
    JJ::Int64 = -1
    PropagatorKernel::Int64 = 0 # 0 - vanilla, 1 - optimized, 2 - reference 
end

"""
    MPIAcousticSource

The input is the global index. 

`srci`, `srcj`: nsrc 
`srcv`: nt × nsrc, each row corresponds to a source time function 
"""
mutable struct MPIAcousticSource
    srci::Union{Missing, PyObject}
    srcj::Union{Missing, PyObject}
    srcv::Union{Missing, PyObject}
    function MPIAcousticSource(param::MPIAcousticPropagatorParams, srci::Array{Int64,1}, srcj::Array{Int64,1}, srcv::Union{Array{Float64,2}, PyObject})
        II, JJ = param.II, param.JJ 
        nsrc = length(srci)
        srcv = constant(srcv)
        srcv = srcv'
        local_srci = Int64[]
        local_srcj = Int64[]
        local_srcv = PyObject[]
        for i = 1:nsrc
            if ((II - 1) * param.n + 1 <= srci[i] <= II * param.n) && 
                    ((JJ - 1) * param.n + 1 <= srcj[i] <= JJ * param.n)
                    push!(local_srci, srci[i] - (II - 1) * param.n)
                    push!(local_srcj, srcj[i] - (JJ - 1) * param.n)
                    push!(local_srcv, srcv[i])
            end
        end
        local_srcv = hcat(local_srcv...)
        if length(local_srci)>0
            local_srci, local_srcj, local_srcv = convert_to_tensor([local_srci, local_srcj, local_srcv], [Int64, Int64, Float64])
            new(local_srci, local_srcj, local_srcv)
        else 
            new(missing, missing, missing)
        end
    end
end

mutable struct MPIAcousticReceiver
    rcvi::Union{Missing,Array{Int64, 1}}
    rcvj::Union{Missing,Array{Int64, 1}}
    rcvv::Union{Missing, PyObject}
    function MPIAcousticReceiver(param::MPIAcousticPropagatorParams, rcvi::Array{Int64,1}, rcvj::Array{Int64,1})
        II, JJ = param.II, param.JJ 
        nrcv = length(rcvi)
        local_rcvi = Int64[]
        local_rcvj = Int64[]
        for i = 1:nrcv
            if ((II - 1) * param.n + 1 <= rcvi[i] <= II * param.n) && 
                    ((JJ - 1) * param.n + 1 <= rcvj[i] <= JJ * param.n)
                    push!(local_rcvi, rcvi[i] - (II - 1) * param.n)
                    push!(local_rcvj, rcvj[i] - (JJ - 1) * param.n)
            end
        end
        if length(local_rcvi)>0
            new(local_rcvi, local_rcvj, missing)
        else 
            new(missing, missing, missing)
        end
    end
end

"""
    AcousticPropagator

A structure holding all the information for acoustic simulation
"""
mutable struct MPIAcousticPropagator
    param::MPIAcousticPropagatorParams
    src::MPIAcousticSource
    u::PyObject
    # auxillary variables for PML
    φ::PyObject
    ψ::PyObject
end


function pml_helper(x::Float64, y::Float64, param::MPIAcousticPropagatorParams)
    outx = 0.0; outy = 0.0
    ξx = param.damping_x
    Lx = param.NPOINTS_PML * param.DELTAX
    if x<Lx && param.USE_PML_XMIN
        d = abs(Lx-x)
        outx = ξx * (d/Lx - sin(2π*d/Lx)/(2π))
    elseif x>param.DELTAX*(param.NX+1)-Lx && param.USE_PML_XMAX
        d = abs(x-(param.DELTAX*(param.NX+1)-Lx))
        outx = ξx * (d/Lx - sin(2π*d/Lx)/(2π))
    end

    ξy = param.damping_y
    Ly = param.NPOINTS_PML * param.DELTAY
    if y<Ly && param.USE_PML_YMIN
        d = abs(Ly-y)
        outy = ξy * (d/Ly - sin(2π*d/Ly)/(2π))
    elseif y>param.DELTAY*(param.NY+1)-Ly && param.USE_PML_YMAX
        d = abs(y-(param.DELTAY*(param.NY+1)-Ly))
        outy = ξy * (d/Ly - sin(2π*d/Ly)/(2π))
    end
    
    return outx, outy
end


function compute_PML_Params!(param::MPIAcousticPropagatorParams)
    NX, NY = param.NX, param.NY
    n = param.n
    r = mpi_rank()
    param.M = div(NX,n)
    param.N = div(NY,n)
    param.II, param.JJ = div(r, param.N)+1, mod(r, param.N)+1
    @assert mpi_size() == param.M * param.N
    @assert NX>0 && NY>0
    @assert mod(NX, n)==0 && mod(NY, n)==0
    

    # computing damping coefficient
    c, R = param.vp_ref, param.Rcoef
    Lx = param.NPOINTS_PML * param.DELTAX
    Ly = param.NPOINTS_PML * param.DELTAY
    param.damping_x = c/Lx*log(1/R)
    param.damping_y = c/Ly*log(1/R)
    # @show c, Lx, log(1/R), param.damping_x, param.damping_y

   
    X = (0:n+1)*param.DELTAX .+ n*param.DELTAX*(param.II-1)
    Y = (0:n+1)*param.DELTAY .+ n*param.DELTAY*(param.JJ-1)
    Σx = zeros(n+2, n+2)
    Σy = zeros(n+2, n+2)
    for i = 1:n+2
        for j = 1:n+2
            Σx[i,j], Σy[i,j] = pml_helper(X[i], Y[j], param)
        end
    end

    param.Σx = Σx
    param.Σy = Σy
    param.IJ = get_mpi_id(2:n+1, 2:n+1, n)
    param.IJn = get_mpi_id(2:n+1, 1:n, n)
    param.IJp = get_mpi_id(2:n+1, 3:n+2,  n)
    param.IpJ = get_mpi_id(3:n+2, 2:n+1, n)
    param.IpJp = get_mpi_id(3:n+2, 3:n+2, n)
    param.IpJn = get_mpi_id(3:n+2, 1:n, n)
    param.InJ = get_mpi_id(1:n, 2:n+1, n)
    param.InJn = get_mpi_id(1:n, 1:n, n)
    param.InJp = get_mpi_id(1:n, 3:n+2, n)
    return param
end

function get_mpi_id(a, b, n)
    idx = Int64[]
    for i = 1:length(a)
        for j = 1:length(b)
            push!(idx, (n+2)*(a[i]-1)+b[j])
        end
    end
    idx
end


function one_step(param::MPIAcousticPropagatorParams, w::PyObject, wold::PyObject, 
        φ, ψ, σ::PyObject, τ::PyObject, c::PyObject, i::PyObject)
    n = param.n 
    Δt = param.DELTAT
    hx, hy = param.DELTAX, param.DELTAY
    IJ, IpJ, InJ, IJp, IJn, IpJp, IpJn, InJp, InJn =
        param.IJ, param.IpJ, param.InJ, param.IJp, param.IJn, param.IpJp, param.IpJn, param.InJp, param.InJn
    
    c = reshape(c, (-1,))
    w = reshape(mpi_halo_exchange(w, param.M, param.N, tag = 5*i, deps=cast(Float64, i)), (-1,))
    wold = reshape(mpi_halo_exchange(wold, param.M, param.N, tag = 5*i+1, deps=w[1]), (-1,))
    φ = reshape(mpi_halo_exchange(φ, param.M, param.N, tag = 5*i+2, deps=wold[1]), (-1,))
    ψ = reshape(mpi_halo_exchange(ψ, param.M, param.N, tag = 5*i+3, deps=φ[1]), (-1,))

    u = (2 - σ[IJ]*τ[IJ]*Δt^2 - 2*Δt^2/hx^2 * c - 2*Δt^2/hy^2 * c) * w[IJ] +
            c * (Δt/hx)^2  *  (w[IpJ]+w[InJ]) +
            c * (Δt/hy)^2  *  (w[IJp]+w[IJn]) +
            (Δt^2/(2hx))*(φ[IpJ]-φ[InJ]) +
            (Δt^2/(2hy))*(ψ[IJp]-ψ[IJn]) -
            (1 - (σ[IJ]+τ[IJ])*Δt/2) * wold[IJ] 
    u_local = u / (1 + (σ[IJ]+τ[IJ])/2*Δt)

    u_local_nxn = reshape(u_local, (n, n))
    
    u = reshape(mpi_halo_exchange(u_local_nxn, param.M, param.N, tag = 5*i+4, deps=ψ[1]), (-1,))
    φ = (1. -Δt*σ[IJ]) * φ[IJ] + Δt * c * (τ[IJ] -σ[IJ])/2hx *  
        (u[IpJ]-u[InJ])
    ψ = (1. -Δt*τ[IJ]) * ψ[IJ] + Δt * c * (σ[IJ] -τ[IJ])/2hy * 
        (u[IJp]-u[IJn])
    φ = reshape(φ, (n, n))
    ψ = reshape(ψ, (n, n))
    u = reshape(u[IJ], (n, n))
    
    u, φ, ψ
end


function one_step_ref(param::MPIAcousticPropagatorParams, w::PyObject, wold::PyObject, 
    φ, ψ, σ::PyObject, τ::PyObject, c::PyObject, i::PyObject)
    n = param.n 
    Δt = param.DELTAT
    hx, hy = param.DELTAX, param.DELTAY
    IJ, IpJ, InJ, IJp, IJn, IpJp, IpJn, InJp, InJn =
        param.IJ, param.IpJ, param.InJ, param.IJp, param.IJn, param.IpJp, param.IpJn, param.InJp, param.InJn

    c = reshape(c, (-1,))
    w = reshape(mpi_halo_exchange(w, param.M, param.N, tag = 5*i, deps=cast(Float64, i)), (-1,))
    wold = reshape(mpi_halo_exchange(wold, param.M, param.N, tag = 5*i+1, deps=w[1]), (-1,))
    φ = reshape(mpi_halo_exchange(φ, param.M, param.N, tag = 5*i+2, deps=wold[1]), (-1,))
    ψ = reshape(mpi_halo_exchange(ψ, param.M, param.N, tag = 5*i+3, deps=φ[1]), (-1,))

    u = (2 - σ[IJ]*τ[IJ]*Δt^2 - 2*Δt^2/hx^2 * c - 2*Δt^2/hy^2 * c) * w[IJ] +
            c * (Δt/hx)^2  *  (w[IpJ]+w[InJ]) +
            c * (Δt/hy)^2  *  (w[IJp]+w[IJn]) +
            (Δt^2/(2hx))*(φ[IpJ]-φ[InJ]) +
            (Δt^2/(2hy))*(ψ[IJp]-ψ[IJn]) -
            (1 - (σ[IJ]+τ[IJ])*Δt/2) * wold[IJ] 
    u = u / (1 + (σ[IJ]+τ[IJ])/2*Δt)
    φ = (1. -Δt*σ[IJ]) * φ[IJ] + Δt * c * (τ[IJ] -σ[IJ])/2hx *  (w[IpJ]-w[InJ])
    ψ = (1. -Δt*τ[IJ]) * ψ[IJ] + Δt * c * (σ[IJ] -τ[IJ])/2hy * (w[IJp]-w[IJn])


    φ = reshape(φ, (n, n))
    ψ = reshape(ψ, (n, n))
    u = reshape(u, (n, n))
    u, φ, ψ
end


"""
    MPIAcousticPropagatorSolver(param::MPIAcousticPropagatorParams, src::MPIAcousticSource, c::Union{PyObject, Array{Float64, 2}})

`MPIAcousticPropagatorSolver` is a MPI-enabled solver. The computational domain has a size `NX x NY` and is divided into `M x N`  
"""
function MPIAcousticPropagatorSolver(param::MPIAcousticPropagatorParams, src::MPIAcousticSource, c::Union{PyObject, Array{Float64, 2}})

    c = constant(c)
    n = param.n 

    @assert size(c, 1)==size(c,2)==n 
    if !ismissing(src.srcv)
        @assert size(src.srcv, 2)==length(src.srci)==length(src.srcj)
        @assert size(src.srcv, 1)==param.NSTEP
    end
    

    σij = constant(param.Σx) # (n+2) × (n+2)
    τij = constant(param.Σy) # (n+2) × (n+2)

    σij = reshape(σij, (-1,))
    τij = reshape(τij, (-1,))

    function condition(i, ta, tφ, tψ)
        i<=param.NSTEP+1
    end

    function body(i, ta, tφ, tψ)
        
        if param.PropagatorKernel==0
            one_step_ = one_step 
        elseif param.PropagatorKernel==1
            one_step_ = one_step_ref 
        else
            error("Not implemented")
        end
        u, φ, ψ = one_step_(param, read(ta, i-1), read(ta, i-2), read(tφ, i-1), read(tψ, i-1), σij, τij, c, i)
        srci, srcj, srcv = AcousticSourceAtTimeT(src, i-1)

        if param.IT_DISPLAY>0
            op = tf.cond(tf.equal(tf.math.floormod(i,param.IT_DISPLAY),1), ()->tf.print("time step = ", i-1, " out of $(param.NSTEP)"), ()->tf.no_op())
            i = bind(i, op)
        end

        if !ismissing(srci)
            src_index = (srci - 1) * n + srcj 
            u = reshape(u, (-1,))
            u = scatter_add_op(u, src_index, srcv*param.DELTAT^2)
            u = reshape(u, (n, n))
        end

        ta_, tφ_, tψ_ = write(ta, i, u), write(tφ, i, φ), write(tψ, i, ψ)
        i+1, ta_, tφ_, tψ_
    end

    tu = TensorArray(param.NSTEP+1; clear_after_read=false)
    tφ = TensorArray(param.NSTEP+1; clear_after_read=true)
    tψ = TensorArray(param.NSTEP+1; clear_after_read=true)
    tu = write(tu, 1, constant(zeros(n, n)))
    tφ = write(tφ, 1, constant(zeros(n, n)))
    tψ = write(tψ, 1, constant(zeros(n, n)))
    tu = write(tu, 2, constant(zeros(n, n)))
    tφ = write(tφ, 2, constant(zeros(n, n)))
    tψ = write(tψ, 2, constant(zeros(n, n)))
    i = constant(3, dtype=Int32)
    _, tu, tφ, tψ = while_loop(condition, body, [i,tu,tφ,tψ]; parallel_iterations=1)
    tu = set_shape(stack(tu), (param.NSTEP+1, n, n))
    tφ = set_shape(stack(tφ), (param.NSTEP+1, n, n))
    tψ = set_shape(stack(tψ), (param.NSTEP+1, n, n))

    MPIAcousticPropagator(param, src, tu, tφ, tψ)
end


function AcousticSourceAtTimeT(src::MPIAcousticSource,i::PyObject)
    if ismissing(src.srci)
        return missing, missing, missing 
    else
        return src.srci, src.srcj, src.srcv[i]
    end
end


"""
    SimulatedObservation!(ap::AcousticPropagator, rcv::AcousticReceiver)

Extract and save simulated displacement u into rcv::AcousticReceiver.
"""
function MPISimulatedObservation!(ap::MPIAcousticPropagator, rcv::MPIAcousticReceiver)
    if ismissing(rcv.rcvi)
        op = group(ap.u)
        rcv.rcvv = op
        return 
    end
    idx = @. (rcv.rcvi - 1) *  ap.param.n + rcv.rcvj 
    @assert size(ap.u,1)==ap.param.NSTEP+1
    @assert size(ap.u,2)==ap.param.n
    rcv.rcvv = reshape(ap.u, (ap.param.NSTEP+1, -1))[:, idx]
end