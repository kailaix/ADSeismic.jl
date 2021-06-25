export MPIElasticPropagatorParams, MPIElasticSource, MPIElasticReceiver, MPIElasticPropagator, compute_PML_Params!,
        MPIElasticPropagatorSolver, MPISimulatedObservation!, MPIElasticReceiver, extract_local_patch
@with_kw mutable struct MPIElasticPropagatorParams
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
    f0::Float64 = 5. # source frequency
    vp_ref::Float64 = 2000. 
    USE_PML_XMIN::Bool = true
    USE_PML_XMAX::Bool = true
    USE_PML_YMIN::Bool = true
    USE_PML_YMAX::Bool = true
    NPOINTS_PML::Int64 = 12
    NPOWER::Float64 = 2.
    K_MAX_PML::Float64 = 1.
    ALPHA_MAX_PML::Float64 = 2. * π * ( f0/2.) 
    Rcoef::Float64 = 0.001 # Relative reflection coefficient

    # Auxilliary Data
    ax::Union{Missing, Array{Float64, 2}} = missing
    bx::Union{Missing, Array{Float64, 2}} = missing
    kx::Union{Missing, Array{Float64, 2}} = missing
    ay::Union{Missing, Array{Float64, 2}} = missing
    by::Union{Missing, Array{Float64, 2}} = missing
    ky::Union{Missing, Array{Float64, 2}} = missing
    
    # display params
    IT_DISPLAY::Int64 = 0

    # mpi 
    II::Int64 = -1 
    JJ::Int64 = -1

    # indices 
    i_1j::Union{Missing, Array{Int64, 1}} = missing
    i_2j::Union{Missing, Array{Int64, 1}} = missing
    i1j::Union{Missing, Array{Int64, 1}} = missing
    i2j::Union{Missing, Array{Int64, 1}} = missing
    ij_1::Union{Missing, Array{Int64, 1}} = missing
    ij_2::Union{Missing, Array{Int64, 1}} = missing
    ij1::Union{Missing, Array{Int64, 1}} = missing
    ij2::Union{Missing, Array{Int64, 1}} = missing
    ij::Union{Missing, Array{Int64, 1}} = missing
    i1j1::Union{Missing, Array{Int64, 1}} = missing
    
end

"""
    MPIElasticSource

The input is the global index. 

`srci`, `srcj`: nsrc 
`srcv`: nt × nsrc, each row corresponds to a source time function 
"""
mutable struct MPIElasticSource
    srci::Union{Missing, PyObject}
    srcj::Union{Missing, PyObject}
    srctype::Union{Missing,PyObject}
    srcv::Union{Missing, PyObject}
    function MPIElasticSource(param::MPIElasticPropagatorParams, srci::Array{Int64,1}, 
         srcj::Array{Int64,1}, srctype::Array{Int64, 1}, srcv::Union{Array{Float64,2}, PyObject})
        II, JJ = param.II, param.JJ 
        nsrc = length(srci)
        srcv = constant(srcv)
        srcv = srcv'
        local_srci = Int64[]
        local_srcj = Int64[]
        local_srcv = PyObject[]
        local_srctype = Int64[]
        for i = 1:nsrc
            if ((II - 1) * param.n + 1 <= srci[i] <= II * param.n) && 
                    ((JJ - 1) * param.n + 1 <= srcj[i] <= JJ * param.n)
                    push!(local_srci, srci[i] - (II - 1) * param.n + 2)
                    push!(local_srcj, srcj[i] - (JJ - 1) * param.n + 2)
                    push!(local_srcv, srcv[i])
                    push!(local_srctype, srctype[i])
            end
        end
        local_srcv = hcat(local_srcv...)
        if length(local_srci)>0
            local_srci, local_srcj, local_srctype, local_srcv = convert_to_tensor([local_srci, local_srcj, local_srctype, local_srcv], [Int64, Int64, Int64, Float64])
            new(local_srci, local_srcj, local_srctype, local_srcv)
        else 
            new(missing, missing, missing, missing)
        end
    end
end

mutable struct MPIElasticReceiver
    rcvi::Union{Missing,Array{Int64, 1}}
    rcvj::Union{Missing,Array{Int64, 1}}
    rcvtype::Union{Missing, Array{Int64, 1}}
    rcvv::Union{Missing, PyObject}
    function MPIElasticReceiver(param::MPIElasticPropagatorParams, rcvi::Array{Int64,1}, 
         rcvj::Array{Int64,1}, rcvtype::Array{Int64, 1})
        II, JJ = param.II, param.JJ 
        nrcv = length(rcvi)
        local_rcvi = Int64[]
        local_rcvj = Int64[]
        local_rcvtype = Int64[]
        for i = 1:nrcv
            if ((II - 1) * param.n + 1 <= rcvi[i] <= II * param.n) && 
                    ((JJ - 1) * param.n + 1 <= rcvj[i] <= JJ * param.n)
                    push!(local_rcvi, rcvi[i] - (II - 1) * param.n)
                    push!(local_rcvj, rcvj[i] - (JJ - 1) * param.n)
                    push!(local_rcvtype, rcvtype[i])
            end
        end
        if length(local_rcvi)>0
            new(local_rcvi, local_rcvj, local_rcvtype, missing)
        else 
            new(missing, missing, missing, missing)
        end
    end
end

"""
    ElasticPropagator

A structure holding all the information for acoustic simulation
"""
mutable struct MPIElasticPropagator
    param::MPIElasticPropagatorParams
    src::MPIElasticSource
    vx::PyObject
    vy::PyObject
    sigmaxx::PyObject
    sigmayy::PyObject
    sigmaxy::PyObject
    mem::PyObject
end


function get_mpi_id2(a, b, n)
    idx = Int64[]
    for i = 1:length(a)
        for j = 1:length(b)
            push!(idx, (n+4)*(a[i]-1)+b[j])
        end
    end
    idx
end

function extract_local_patch(param::MPIElasticPropagatorParams, v::Union{PyObject, Array{Float64,2}}; deps = missing, tag = 0)
    v = v[(param.II - 1) * param.n + 1:param.II * param.n, (param.JJ - 1) * param.n + 1:param.JJ * param.n]
    v = mpi_halo_exchange2(v, param.M, param.N, deps = deps, tag = tag)
end

function compute_PML_Params!(param::MPIElasticPropagatorParams)
    NX, NY = param.NX, param.NY
    n = param.n
    r = mpi_rank()
    param.M = div(NX,n)
    param.N = div(NY,n)
    param.II, param.JJ = div(r, param.N)+1, mod(r, param.N)+1
    if mpi_size() != param.M * param.N
        error("MPI Size Mismatch: mpi_size = $(mpi_size()), M = $(param.M), N = $(param.N)")
    end
    @assert NX>0 && NY>0
    @assert mod(NX, n)==0 && mod(NY, n)==0

    # ID 
    k1 = 4:n+3
    k2 = 5:n+4
    k_1 = 2:n+1
    k_2 = 1:n
    kk = 3:n+2
    param.i_1j = get_mpi_id2(k_1, kk, n)
    param.i_2j = get_mpi_id2(k_2, kk, n)
    param.i1j = get_mpi_id2(k1, kk, n)
    param.i2j = get_mpi_id2(k2, kk, n)
    param.ij_1 = get_mpi_id2(kk, k_1, n)
    param.ij_2 = get_mpi_id2(kk, k_2, n)
    param.ij1 = get_mpi_id2(kk, k1, n)
    param.ij2 = get_mpi_id2(kk, k2, n)
    param.ij = get_mpi_id2(kk, kk, n)
    param.i1j1 = get_mpi_id2(k1, k1, n)
    

    ## define profile of absorption in PML region
    NPOINTS_PML = param.NPOINTS_PML
    NPOWER = param.NPOWER
    NX = param.NX
    NY = param.NY
    DELTAX = param.DELTAX
    DELTAY = param.DELTAY
    DELTAT = param.DELTAT
    K_MAX_PML = param.K_MAX_PML
    ALPHA_MAX_PML = param.ALPHA_MAX_PML
    USE_PML_XMIN = param.USE_PML_XMIN
    USE_PML_XMAX = param.USE_PML_XMAX
    USE_PML_YMIN = param.USE_PML_YMIN
    USE_PML_YMAX = param.USE_PML_YMAX
    Rcoef = param.Rcoef
    
    thickness_PML_x = NPOINTS_PML * DELTAX
    thickness_PML_y = NPOINTS_PML * DELTAY
    
    # reflection coefficient
    d0_x = - (NPOWER + 1) * param.vp_ref * log(Rcoef) / (2. * thickness_PML_x)
    d0_y = - (NPOWER + 1) * param.vp_ref * log(Rcoef) / (2. * thickness_PML_y)
    
    xoriginleft = thickness_PML_x
    xoriginright = (NX-0.5)*DELTAX - thickness_PML_x
    
    a_x = zeros(n)
    a_y = zeros(n)
    a_x_half = zeros(n)
    a_y_half = zeros(n)
    b_x = zeros(n)
    b_y = zeros(n)
    b_x_half = zeros(n)
    b_y_half = zeros(n)
    d_x = zeros(n)
    d_y = zeros(n)
    d_x_half = zeros(n)
    d_y_half = zeros(n)
    K_x = ones(n)
    K_y = ones(n)
    K_x_half = ones(n)
    K_y_half = ones(n)
    alpha_x = zeros(n)
    alpha_y = zeros(n)
    alpha_x_half = zeros(n)
    alpha_y_half = zeros(n)
    
    for i = 1:n
        xval = DELTAX * (i-1) + (param.II - 1) * param.n * DELTAX
        if (USE_PML_XMIN)
            
            abscissa_in_PML = xoriginleft - xval
            if (abscissa_in_PML >= 0.0)
                abscissa_normalized = abscissa_in_PML / thickness_PML_x
                d_x[i] = d0_x * abscissa_normalized^NPOWER
                K_x[i] = 1. + (K_MAX_PML - 1.) * abscissa_normalized^NPOWER
                # alpha_x[i] = ALPHA_MAX_PML * (1. - abscissa_normalized)
                alpha_x[i] = ALPHA_MAX_PML * abscissa_normalized
            end
            
            abscissa_in_PML = xoriginleft - (xval + DELTAX/2.)
            if (abscissa_in_PML >= 0.0)
                abscissa_normalized = abscissa_in_PML / thickness_PML_x
                d_x_half[i] = d0_x * abscissa_normalized^NPOWER
                K_x_half[i] = 1. + (K_MAX_PML - 1.) * abscissa_normalized^NPOWER
                # alpha_x_half[i] = ALPHA_MAX_PML * (1. - abscissa_normalized)
                alpha_x_half[i] = ALPHA_MAX_PML * abscissa_normalized
            end
            
        end
        
        if (USE_PML_XMAX)
            abscissa_in_PML = xval - xoriginright
            if (abscissa_in_PML >= 0)
                abscissa_normalized = abscissa_in_PML / thickness_PML_x
                d_x[i] = d0_x * abscissa_normalized^NPOWER
                K_x[i] = 1. + (K_MAX_PML - 1.) * abscissa_normalized^NPOWER
                # alpha_x[i] = ALPHA_MAX_PML * (1. - abscissa_normalized)
                alpha_x[i] = ALPHA_MAX_PML * abscissa_normalized
            end
            
            abscissa_in_PML = xval + DELTAX/2.0 - xoriginright
            if (abscissa_in_PML >= 0)
                abscissa_normalized = abscissa_in_PML / thickness_PML_x
                d_x_half[i] = d0_x * abscissa_normalized^NPOWER
                K_x_half[i] = 1. + (K_MAX_PML - 1.) * abscissa_normalized^NPOWER
                # alpha_x_half[i] = ALPHA_MAX_PML * (1. - abscissa_normalized)
                alpha_x_half[i] = ALPHA_MAX_PML * abscissa_normalized
            end
        end
        
        if (alpha_x[i] < 0) 
            alpha_x[i] = 0
        end
        if (alpha_x_half[i] < 0) 
            alpha_x_half[i] = 0
        end
        
        b_x[i] = exp(- (d_x[i] / K_x[i] + alpha_x[i]) * DELTAT)
        b_x_half[i] = exp(- (d_x_half[i] / K_x_half[i] + alpha_x_half[i]) * DELTAT)
        
        if (abs(d_x[i]) > 1e-6) 
            a_x[i] = d_x[i] * (b_x[i] - 1.) / (K_x[i] * (d_x[i] + K_x[i] * alpha_x[i]))
        end
        if (abs(d_x_half[i]) > 1e-6) 
            a_x_half[i] = d_x_half[i] * (b_x_half[i] - 1.) / (K_x_half[i] * (d_x_half[i] + K_x_half[i] * alpha_x_half[i]))
        end
    end
    
    yoriginbottom = thickness_PML_y
    yorigintop = (NY-0.5)*DELTAY - thickness_PML_y
    
    for j = 1:n
        
        yval = DELTAY * (j-1) + (param.JJ - 1) * param.n * DELTAX
        if param.USE_PML_YMIN
            abscissa_in_PML = yoriginbottom - yval
            if (abscissa_in_PML >= 0.0)
                abscissa_normalized = abscissa_in_PML / thickness_PML_y
                d_y[j] = d0_y * abscissa_normalized^NPOWER
                K_y[j] = 1. + (K_MAX_PML - 1.) * abscissa_normalized^NPOWER
                # alpha_y[j] = ALPHA_MAX_PML * (1. - abscissa_normalized)
                alpha_y[j] = ALPHA_MAX_PML * abscissa_normalized
            end
            abscissa_in_PML = yoriginbottom - (yval + DELTAY/2.)
            if abscissa_in_PML >= 0
                abscissa_normalized = abscissa_in_PML / thickness_PML_y
                d_y_half[j] = d0_y * abscissa_normalized^NPOWER
                K_y_half[j] = 1. + (K_MAX_PML - 1.) * abscissa_normalized^NPOWER
                # alpha_y_half[j] = ALPHA_MAX_PML * (1. - abscissa_normalized)
                alpha_y_half[j] = ALPHA_MAX_PML * abscissa_normalized
            end
        end
        
        if param.USE_PML_YMAX
            abscissa_in_PML = yval - yorigintop
            if abscissa_in_PML >= 0
                abscissa_normalized = abscissa_in_PML / thickness_PML_y
                d_y[j] = d0_y * abscissa_normalized^NPOWER
                K_y[j] = 1. + (K_MAX_PML - 1.) * abscissa_normalized^NPOWER
                # alpha_y[j] = ALPHA_MAX_PML * (1. - abscissa_normalized)
                alpha_y[j] = ALPHA_MAX_PML * abscissa_normalized
            end
            
            abscissa_in_PML = yval + DELTAY/2. - yorigintop
            if abscissa_in_PML >= 0
                abscissa_normalized = abscissa_in_PML / thickness_PML_y
                d_y_half[j] = d0_y * abscissa_normalized^NPOWER
                K_y_half[j] = 1. + (K_MAX_PML - 1.) * abscissa_normalized^NPOWER
                # alpha_y_half[j] = ALPHA_MAX_PML * (1. - abscissa_normalized)
                alpha_y_half[j] = ALPHA_MAX_PML * abscissa_normalized
            end
        end
        
        b_y[j] = exp(- (d_y[j] / K_y[j] + alpha_y[j]) * DELTAT)
        b_y_half[j] = exp(- (d_y_half[j] / K_y_half[j] + alpha_y_half[j]) * DELTAT)
        if abs(d_y[j]) > 1e-6
            a_y[j] = d_y[j] * (b_y[j] - 1.) / (K_y[j] * (d_y[j] + K_y[j] * alpha_y[j]))
        end
        if (abs(d_y_half[j]) > 1e-6) 
            a_y_half[j] = d_y_half[j] * (b_y_half[j] - 1.) / (K_y_half[j] * (d_y_half[j] + K_y_half[j] * alpha_y_half[j]))
        end
    end
    # @show norm(d_x-d_y)

    ax = [a_x'; a_x_half']
    bx = [b_x'; b_x_half']
    kx = [K_x'; K_x_half']
    ay = [a_y'; a_y_half']
    by = [b_y'; b_y_half']
    ky = [K_y'; K_y_half']
    
    
    param.ax = ax
    param.bx = bx
    param.kx = kx
    param.ay = ay
    param.by = by
    param.ky = ky
    param
end

function MPIElasticPropagatorSolver(param::MPIElasticPropagatorParams, src::MPIElasticSource, 
    ρ::Union{PyObject, Array{Float64, 2}}, λ::Union{PyObject, Array{Float64, 2}}, μ::Union{PyObject, Array{Float64, 2}};
    tag_offset::Int64 = 0, dep = nothing)
    ρ = reshape(convert_to_tensor(ρ),(-1,))
    λ = reshape(convert_to_tensor(λ),(-1,))
    μ = reshape(convert_to_tensor(μ),(-1,))
    n = param.n
    @assert size(ρ)==size(λ)==size(μ)==((n+4)^2,)
    @assert param.II>0
    if !ismissing(src.srcv)
        @assert size(src.srcv, 2)==length(src.srci)==length(src.srcj)==length(src.srctype)
        @assert size(src.srcv, 1)==param.NSTEP
    end
    NSTEP = param.NSTEP
    vx_arr = TensorArray(NSTEP+1); 
    vx_arr = write(vx_arr, 1, constant(zeros(n^2)))
    vy_arr = TensorArray(NSTEP+1); 
    vy_arr = write(vy_arr, 1, constant(zeros(n^2)))
    sigmaxx_arr = TensorArray(NSTEP+1); 
    sigmaxx_arr = write(sigmaxx_arr, 1, constant(zeros(n^2)))
    sigmayy_arr = TensorArray(NSTEP+1); 
    sigmayy_arr = write(sigmayy_arr, 1, constant(zeros(n^2)))
    sigmaxy_arr = TensorArray(NSTEP+1); 
    sigmaxy_arr = write(sigmaxy_arr, 1, constant(zeros(n^2)))
    mem_arr = [TensorArray(NSTEP+1) for i = 1:8] 
    for i = 1:8
        mem_arr[i] = write(mem_arr[i], 1, constant(zeros(n^2)))
    end
    ax,bx,kx,ay,by,ky = param.ax,param.bx,param.kx,param.ay,param.by,param.ky 
    function condition(i, vx_arr, vy_arr, sigmaxx_arr, sigmayy_arr, sigmaxy_arr, mem_arr...)
        i<=NSTEP+1
    end
    
    if isnothing(dep)
        dep = 0.0
    end

    function body(i, vx_arr, vy_arr, sigmaxx_arr, sigmayy_arr, sigmaxy_arr, mem_arr...)
        mem = Array{PyObject}(undef, 8)
        σxx,σyy,σxy,vx,vy = 
            read(sigmaxx_arr, i-1), read(sigmayy_arr, i-1), read(sigmaxy_arr, i-1), 
            read(vx_arr, i-1), read(vy_arr, i-1)
        for k = 1:8
            m = read(mem_arr[k], i-1)
            mem[k] = _reshape_and_halo_exchange(m, param, 18*i+k-1+tag_offset, k>1 ? mem[k-1][1] + dep : λ[1] + ρ[1] + μ[1] + vy[1] + σxy[1] + dep)
        end

        mpi_idx = 18*i+8+tag_offset
        
        halo = (op, deps, mpi_idx)->begin 
            op = _reshape_and_halo_exchange(op, param, mpi_idx, deps)
            mpi_idx = mpi_idx + 1
            op, mpi_idx
        end
        vx, mpi_idx = halo(vx, mem[8][1], mpi_idx)
        vy, mpi_idx = halo(vy, vx[1], mpi_idx)
        σxx, mpi_idx = halo(σxx, vy[1], mpi_idx)
        σyy, mpi_idx = halo(σyy, σxx[1], mpi_idx)
        σxy, mpi_idx = halo(σxy, σyy[1], mpi_idx)

        srci,srcj,srctype,srcv = ElasticSourceAtTimeT(src, i-1) 
        σxx, σyy, σxy,vx, vy, mem, mpi_idx = one_step(param, ρ, λ, μ, vx, vy, σxx, σyy, σxy,mem,
            ax, bx, ay, by, kx, ky, srci, srcj, srcv, srctype, mpi_idx)

        sigmaxx_arr,sigmayy_arr,sigmaxy_arr,vx_arr,vy_arr = 
            write(sigmaxx_arr, i, σxx), write(sigmayy_arr, i, σyy), write(sigmaxy_arr, i, σxy),
                    write(vx_arr, i, vx), write(vy_arr, i, vy)
        
        mem_arr_ = Array{PyObject}(undef, 8)
        for k = 1:8
            mem_arr_[k] = write(mem_arr[k], i, mem[k])
        end

        if param.IT_DISPLAY>0
            op = tf.cond(tf.equal(tf.math.floormod(i,param.IT_DISPLAY),1), ()->tf.print("time step = ", i-1, " out of $NSTEP"), ()->tf.no_op())
            i = bind(i, op)
        end

        i+1, vx_arr, vy_arr, sigmaxx_arr, sigmayy_arr, sigmaxy_arr, mem_arr_...
    end
    i = constant(2, dtype=Int32)
    _, vx_arr, vy_arr, sigmaxx_arr, sigmayy_arr, sigmaxy_arr, mem_arrs = 
        while_loop(condition, body, [i, vx_arr, vy_arr, sigmaxx_arr, sigmayy_arr, sigmaxy_arr, mem_arr...])
    
    vx = reshape(stack(vx_arr), (param.NSTEP+1, n, n)) 
    vy = reshape(stack(vy_arr), (param.NSTEP+1, n, n))
    σxx = reshape(stack(sigmaxx_arr), (param.NSTEP+1, n, n))
    σyy = reshape(stack(sigmayy_arr), (param.NSTEP+1, n, n))
    σxy = reshape(stack(sigmaxy_arr), (param.NSTEP+1, n, n))
    MPIElasticPropagator(
        param, src, vx, vy, σxx, σyy, σxy, mem_arrs
    )
end


function _reshape_and_print(o, m, n, info)
    o = reshape(o, (m, n))    
    o = print_tensor(o, info)
    reshape(o, (m*n,))
end

function _reshape_and_halo_exchange(o, param, tag, deps)
    @assert size(o)==(param.n^2,)
    o = reshape(o, (param.n, param.n))
    out = mpi_halo_exchange2(o, param.M, param.N, tag = tag, deps= deps)
    reshape(out, ((param.n+4)^2,))
end


function fw1(param::MPIElasticPropagatorParams, ρ, λ, μ, vx, vy, σxx, σyy, σxy,mem,
        ax, bx, ay, by, kx, ky, mpi_idx)
    NX, NY = param.NX, param.NY
    n = param.n
    DELTAX, DELTAY, DELTAT = param.DELTAX, param.DELTAY, param.DELTAT
    i_1j = param.i_1j
    i_2j = param.i_2j
    i1j = param.i1j
    i2j = param.i2j
    ij_1 = param.ij_1
    ij_2 = param.ij_2
    ij1 = param.ij1
    ij2 = param.ij2
    ij = param.ij


    # λ_ = 0.5*(λ[i1j]+λ[ij])
    # μ_ = 0.5*(μ[i1j]+μ[ij])
    λ_ = λ[ij]
    μ_ = μ[ij]
    λμ = λ_ + 2μ_
    value_dvx_dx = (27vx[i1j]-27vx[ij]-vx[i2j]+vx[i_1j])/(24*DELTAX)
    value_dvy_dy = (27*vy[ij]-27*vy[ij_1]-vy[ij1]+vy[ij_2]) / (24*DELTAY)
    # broadcast
    mem[1] =  makevector(mem[1], ij, adbroadcast(bx[2,:], mem[1][ij], 1) + adbroadcast(ax[2,:], value_dvx_dx, 1));
    mem[2] =  makevector(mem[2], ij, adbroadcast(by[1,:], mem[2][ij], 2) + adbroadcast(ay[1,:], value_dvy_dy, 2));
    value_dvx_dx = adbroadcast(value_dvx_dx, kx[2,:], 3) + mem[1][ij]
    value_dvy_dy = adbroadcast(value_dvy_dy, ky[1,:], 4) + mem[2][ij]
    σxx = scatter_add_op(σxx, ij, (λμ * value_dvx_dx + λ_ * value_dvy_dy) * DELTAT )
    σyy = scatter_add_op(σyy, ij, (λμ * value_dvy_dy + λ_ * value_dvx_dx) * DELTAT )


    σxx = _reshape_and_halo_exchange(σxx[ij], param, mpi_idx, σxy[1])
    σyy = _reshape_and_halo_exchange(σyy[ij], param, mpi_idx+1, σxx[1])

    return vx, vy, σxx, σyy,σxy, mem, mpi_idx+2
end

function fw2(param::MPIElasticPropagatorParams, ρ, λ, μ, vx, vy, σxx, σyy, σxy,mem,
    ax, bx, ay, by, kx, ky, mpi_idx)
    NX, NY = param.NX, param.NY
    n = param.n
    DELTAX, DELTAY, DELTAT = param.DELTAX, param.DELTAY, param.DELTAT
    i_1j = param.i_1j
    i_2j = param.i_2j
    i1j = param.i1j
    i2j = param.i2j
    ij_1 = param.ij_1
    ij_2 = param.ij_2
    ij1 = param.ij1
    ij2 = param.ij2
    ij = param.ij


    # μ_ = 0.5*(μ[ij]+μ[ij1])
    μ_ = μ[ij]
    value_dvy_dx = (27vy[ij]-27vy[i_1j]-vy[i1j]+vy[i_2j])/(24*DELTAX)
    value_dvx_dy = (27*vx[ij1]-27*vx[ij]-vx[ij2]+vx[ij_1]) / (24*DELTAY)

    # broadcast
    mem[3] = makevector(mem[3], ij,  adbroadcast(bx[1,:], mem[3][ij], 1) + adbroadcast(ax[1, :], value_dvy_dx, 1))
    mem[4] = makevector(mem[4], ij,  adbroadcast(by[2,:], mem[4][ij], 2) + adbroadcast(ay[2, :], value_dvx_dy, 2));

    value_dvy_dx = adbroadcast(value_dvy_dx, kx[1,:], 3) + mem[3][ij]
    value_dvx_dy = adbroadcast(value_dvx_dy, ky[1,:], 4) + mem[4][ij]

    σxy = scatter_add_op(σxy, ij, μ_*(value_dvy_dx + value_dvx_dy) * DELTAT )

    σxy = _reshape_and_halo_exchange(σxy[ij], param, mpi_idx, σyy[1])

    return vx, vy, σxx, σyy, σxy,mem, mpi_idx+1
end

function fw3(param::MPIElasticPropagatorParams, ρ, λ, μ, vx, vy, σxx, σyy, σxy,mem,
    ax, bx, ay, by, kx, ky, mpi_idx)
    NX, NY = param.NX, param.NY
    n = param.n
    DELTAX, DELTAY, DELTAT = param.DELTAX, param.DELTAY, param.DELTAT
    i_1j = param.i_1j
    i_2j = param.i_2j
    i1j = param.i1j
    i2j = param.i2j
    ij_1 = param.ij_1
    ij_2 = param.ij_2
    ij1 = param.ij1
    ij2 = param.ij2
    ij = param.ij

  
    value_dsigmaxx_dx = (27*σxx[ij]-27*σxx[i_1j]-σxx[i1j]+σxx[i_2j]) / (24*DELTAX);
    value_dsigmaxy_dy = (27*σxy[ij]-27*σxy[ij_1]-σxy[ij1]+σxy[ij_2]) / (24*DELTAY);

    mem[5] = makevector(mem[5],ij, adbroadcast(bx[1,:], mem[5][ij], 1) + adbroadcast(ax[1,:], value_dsigmaxx_dx, 1))
    mem[6] = makevector(mem[6],ij, adbroadcast(by[1,:], mem[6][ij], 2) + adbroadcast(ay[1,:], value_dsigmaxy_dy, 2))

    value_dsigmaxx_dx = adbroadcast(value_dsigmaxx_dx, kx[1,:], 3) + mem[5][ij]
    value_dsigmaxy_dy = adbroadcast(value_dsigmaxy_dy, ky[1,:], 4) + mem[6][ij]

    vx = scatter_add_op(vx, ij, (value_dsigmaxx_dx + value_dsigmaxy_dy) * DELTAT / ρ[ij])
    vx = _reshape_and_halo_exchange(vx[ij], param, mpi_idx, σxy[1])

    return vx, vy, σxx, σyy, σxy, mem, mpi_idx+1
end



function fw4(param::MPIElasticPropagatorParams, ρ, λ, μ, vx, vy, σxx, σyy, σxy,mem,
    ax, bx, ay, by, kx, ky, mpi_idx)
    NX, NY, n = param.NX, param.NY, param.n
    DELTAX, DELTAY, DELTAT = param.DELTAX, param.DELTAY, param.DELTAT
    i_1j = param.i_1j
    i_2j = param.i_2j
    i1j = param.i1j
    i2j = param.i2j
    ij_1 = param.ij_1
    ij_2 = param.ij_2
    ij1 = param.ij1
    ij2 = param.ij2
    ij = param.ij
    i1j1 = param.i1j1


    # ρ_ = 0.25 * (ρ[ij] + ρ[i1j] + ρ[i1j1] + ρ[ij1]);
    ρ_ = ρ[ij]

    value_dsigmaxy_dx = (27*σxy[i1j]-27*σxy[ij]-σxy[i2j]+σxy[i_1j]) / (24*DELTAX);
    value_dsigmayy_dy = (27*σyy[ij1]-27*σyy[ij]-σyy[ij2]+σyy[ij_1]) / (24*DELTAY);

    mem[7] = makevector(mem[7], ij, adbroadcast(bx[2,:], mem[7][ij], 1) + adbroadcast(ax[2,:] , value_dsigmaxy_dx, 1))
    mem[8] = makevector(mem[8], ij, adbroadcast(by[2,:], mem[8][ij], 2) + adbroadcast(ay[2,:] , value_dsigmayy_dy, 2))

    value_dsigmaxy_dx = adbroadcast(value_dsigmaxy_dx, kx[2,:], 3) + mem[7][ij]
    value_dsigmayy_dy = adbroadcast(value_dsigmayy_dy, ky[2,:], 4) + mem[8][ij]

    vy = scatter_add_op( vy, ij, (value_dsigmaxy_dx + value_dsigmayy_dy) * DELTAT / ρ_)
    vy = _reshape_and_halo_exchange(vy[ij], param, mpi_idx, vx[1])
    return vx, vy, σxx, σyy, σxy,mem, mpi_idx+1
end

function one_step(param::MPIElasticPropagatorParams, ρ, λ, μ, vx, vy, σxx, σyy, σxy,mem,
    ax, bx, ay, by, kx, ky, srci, srcj, srcv, srctype, mpi_idx)
    n = param.n
    @assert size(ρ)==size(λ)==size(μ)==size(vx)==size(vy)==size(σxx)==size(σyy)==size(σxy)==((n+4)^2,)

    add_source = load_op_and_grad("$(@__DIR__)/../deps/CustomOps/build/libADSeismic","add_source", multiple=true)
    vx, vy, σxx, σyy, σxy,mem, mpi_idx = fw1(param, ρ, λ, μ, vx, vy, σxx, σyy, σxy,mem,
                    ax, bx, ay, by, kx, ky, mpi_idx)
    vx, vy, σxx, σyy, σxy,mem, mpi_idx = fw2(param, ρ, λ, μ, vx, vy, σxx, σyy, σxy,mem,
                    ax, bx, ay, by, kx, ky, mpi_idx)
    vx, vy, σxx, σyy, σxy,mem, mpi_idx = fw3(param, ρ, λ, μ, vx, vy, σxx, σyy, σxy,mem,
                    ax, bx, ay, by, kx, ky, mpi_idx)
    vx, vy, σxx, σyy, σxy,mem, mpi_idx = fw4(param, ρ, λ, μ, vx, vy, σxx, σyy, σxy,mem,
                    ax, bx, ay, by, kx, ky, mpi_idx)    
    if !ismissing(srci)
        σxx, σyy, σxy,vx, vy  =  add_source(σxx, σyy, σxy,vx, vy, constant(srci),constant(srcj),constant(srctype),constant(n+2),constant(n+2),srcv)
    end
    ij = param.ij
    σxx, σyy, σxy,vx, vy  = σxx[ij], σyy[ij], σxy[ij],vx[ij],vy[ij]
    for i = 1:8
        mem[i] = mem[i][ij]
    end
    return σxx, σyy, σxy,vx, vy, mem, mpi_idx
end





function ElasticSourceAtTimeT(src::MPIElasticSource,i::PyObject)
    if ismissing(src.srci)
        return missing, missing, missing, missing
    else
        return src.srci, src.srcj, src.srctype, src.srcv[i]
    end
end


"""
    SimulatedObservation!(ap::ElasticPropagator, rcv::ElasticReceiver)

Extract and save simulated displacement u into rcv::ElasticReceiver.
"""
function MPISimulatedObservation!(ep::MPIElasticPropagator, rcv::MPIElasticReceiver)
    if ismissing(rcv.rcvi)
        op = group([ep.vx, ep.vy, ep.sigmaxx, ep.sigmayy, ep.sigmaxy])
        rcv.rcvv = op
        return 
    end

    N = (ep.param.n)*(ep.param.n)
    get_receive = load_op_and_grad("$(@__DIR__)/../deps/CustomOps/build/libADSeismic", "get_receive")
    u = get_receive(tf.reshape(ep.vx, (ep.param.NSTEP+1, N)), 
                    tf.reshape(ep.vy, (ep.param.NSTEP+1, N)), 
                    tf.reshape(ep.sigmaxx, (ep.param.NSTEP+1, N)), 
                    tf.reshape(ep.sigmayy, (ep.param.NSTEP+1, N)), 
                    tf.reshape(ep.sigmaxy, (ep.param.NSTEP+1, N)),
                    convert_to_tensor(rcv.rcvi), convert_to_tensor(rcv.rcvj), convert_to_tensor(rcv.rcvtype),
                        constant(ep.param.n-2), constant(ep.param.n-2))
    rcv.rcvv = u
end