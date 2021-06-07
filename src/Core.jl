export SimulatedObservation!, ElasticPropagatorSolver, AcousticPropagatorSolver

function gather_ops(ipt,ii)
    gather_ops_ = load_op_and_grad("$(@__DIR__)/../deps/CustomOps/build/libADSeismic","gather_ops")
    ipt,ii = convert_to_tensor(Any[ipt,ii], [Float64,Int64])
    out = gather_ops_(ipt,ii)
    set_shape(out, (length(ii), ))
end

function Base.:getindex(o::PyObject, ii::Array{Int64,1})
    gather_ops(o, ii)
end

function scatter_add_op(ipt, ii, vv)
    scatter_add_ops_ = load_op_and_grad("$(@__DIR__)/../deps/CustomOps/build/libADSeismic","scatter_add_ops")
    ipt,ii,vv = convert_to_tensor(Any[ipt,ii,vv], [Float64,Int64,Float64])
    out = scatter_add_ops_(ipt,ii,vv)
    set_shape(out, (length(ipt), ))
    # ipt = scatter_add(ipt, ii, vv)
end

function scatter_nd_ops(ii,vv,m)
    scatter_nd_ops_ = load_op_and_grad("$(@__DIR__)/../deps/CustomOps/build/libADSeismic","scatter_nd_ops")
    ii,vv,m_ = convert_to_tensor(Any[ii,vv,m], [Int64,Float64,Int64])
    out = scatter_nd_ops_(ii,vv,m_)
    set_shape(out, (m,))
    # vector(ii, vv, m)
end

function ElasticPropagatorSolver(param::ElasticPropagatorParams, src::ElasticSource, 
    ρ::Union{PyObject, Array{Float64, 2}}, λ::Union{PyObject, Array{Float64, 2}}, μ::Union{PyObject, Array{Float64, 2}})
    ρ = tf.reshape(convert_to_tensor(ρ),(-1,))
    λ = tf.reshape(convert_to_tensor(λ),(-1,))
    μ = tf.reshape(convert_to_tensor(μ),(-1,))
    NSTEP = param.NSTEP
    vx_arr = TensorArray(NSTEP+1); vx_arr = write(vx_arr, 1, constant(zeros((param.NX+2)*(param.NY+2))))
    vy_arr = TensorArray(NSTEP+1); vy_arr = write(vy_arr, 1, constant(zeros((param.NX+2)*(param.NY+2))))
    sigmaxx_arr = TensorArray(NSTEP+1); sigmaxx_arr = write(sigmaxx_arr, 1, constant(zeros((param.NX+2)*(param.NY+2))))
    sigmayy_arr = TensorArray(NSTEP+1); sigmayy_arr = write(sigmayy_arr, 1, constant(zeros((param.NX+2)*(param.NY+2))))
    sigmaxy_arr = TensorArray(NSTEP+1); sigmaxy_arr = write(sigmaxy_arr, 1, constant(zeros((param.NX+2)*(param.NY+2))))
    mem_arr = [TensorArray(NSTEP+1) for i = 1:8] 
    for i = 1:8
        mem_arr[i] = write(mem_arr[i], 1, constant(zeros((param.NX+2)*(param.NY+2))))
    end
    ax,bx,kx,ay,by,ky = compute_PML_Params(param)
    
    function condition(i, vx_arr, vy_arr, sigmaxx_arr, sigmayy_arr, sigmaxy_arr, mem_arr...)
        i<=NSTEP+1
    end
    
    function body(i, vx_arr, vy_arr, sigmaxx_arr, sigmayy_arr, sigmaxy_arr, mem_arr...)
        mem = Array{PyObject}(undef, 8)
        σxx,σyy,σxy,vx,vy = 
            read(sigmaxx_arr, i-1), read(sigmayy_arr, i-1), read(sigmaxy_arr, i-1), 
            read(vx_arr, i-1), read(vy_arr, i-1)
        for k = 1:8
            mem[k] = read(mem_arr[k], i-1)
        end
        srci,srcj,srctype,srcv = ElasticSourceAtTimeT(src, i-1) 
        σxx, σyy, σxy,vx, vy, mem = one_step(param, ρ, λ, μ, vx, vy, σxx, σyy, σxy,mem,
            ax, bx, ay, by, kx, ky, srci, srcj, srcv, srctype)

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
    
    vx = tf.reshape(stack(vx_arr), (param.NSTEP+1, param.NX+2, param.NY+2))
    vy = tf.reshape(stack(vy_arr), (param.NSTEP+1, param.NX+2, param.NY+2))
    σxx = tf.reshape(stack(sigmaxx_arr), (param.NSTEP+1, param.NX+2, param.NY+2))
    σyy = tf.reshape(stack(sigmayy_arr), (param.NSTEP+1, param.NX+2, param.NY+2))
    σxy = tf.reshape(stack(sigmaxy_arr), (param.NSTEP+1, param.NX+2, param.NY+2))
    ElasticPropagator(
        param, src, vx, vy, σxx, σyy, σxy, mem_arrs
    )
end

function fw1(param::ElasticPropagatorParams, ρ, λ, μ, vx, vy, σxx, σyy, σxy,mem,
        ax, bx, ay, by, kx, ky)
    NX, NY = param.NX, param.NY
    DELTAX, DELTAY, DELTAT = param.DELTAX, param.DELTAY, param.DELTAT
    i_1j = getid(param, 1:NX-1, 3:NY+1)
    ij = getid(param, 2:NX, 3:NY+1)
    i1j = getid(param, 3:NX+1, 3:NY+1)
    i2j = getid(param, 4:NX+2, 3:NY+1)

    ij_2 = getid(param, 2:NX, 1:NY-1)
    ij_1 = getid(param, 2:NX, 2:NY)
    ij1 = getid(param, 2:NX, 4:NY+2)
    
    λ_ = 0.5*(λ[i1j]+λ[ij])
    μ_ = 0.5*(μ[i1j]+μ[ij])
    λμ = λ_ + 2μ_
    value_dvx_dx = (27vx[i1j]-27vx[ij]-vx[i2j]+vx[i_1j])/(24*DELTAX)
    value_dvy_dy = (27*vy[ij]-27*vy[ij_1]-vy[ij1]+vy[ij_2]) / (24*DELTAY)
    
    # broadcast
    mem[1] =  makevector(mem[1], ij, adbroadcast(bx[2,1:end-1], mem[1][ij], 1) + adbroadcast(ax[2,1:end-1], value_dvx_dx, 1));
    mem[2] =  makevector(mem[2], ij, adbroadcast(by[1,2:end], mem[2][ij], 2) + adbroadcast(ay[1,2:end], value_dvy_dy, 2));

    value_dvx_dx = adbroadcast(value_dvx_dx, kx[2,1:end-1], 3) + mem[1][ij]
    value_dvy_dy = adbroadcast(value_dvy_dy, ky[1,2:end], 4) + mem[2][ij]
    
    σxx = scatter_add_op(σxx, ij, (λμ * value_dvx_dx + λ_ * value_dvy_dy) * DELTAT )
    σyy = scatter_add_op(σyy, ij, (λμ * value_dvy_dy + λ_ * value_dvx_dx) * DELTAT )

    return vx, vy, σxx, σyy,σxy, mem
end

function fw2(param::ElasticPropagatorParams, ρ, λ, μ, vx, vy, σxx, σyy, σxy,mem,
    ax, bx, ay, by, kx, ky)
    NX, NY = param.NX, param.NY
    DELTAX, DELTAY, DELTAT = param.DELTAX, param.DELTAY, param.DELTAT
    i_2j = getid(param, 1:NX-1, 2:NY)
    i_1j = getid(param, 2:NX, 2:NY)
    ij = getid(param, 3:NX+1, 2:NY)
    i1j = getid(param, 4:NX+2, 2:NY)

    ij_1 = getid(param, 3:NX+1, 1:NY-1)
    ij1 = getid(param, 3:NX+1, 3:NY+1)
    ij2 = getid(param, 3:NX+1, 4:NY+2)

    μ_ = 0.5*(μ[ij]+μ[ij1])
    value_dvy_dx = (27vy[ij]-27vy[i_1j]-vy[i1j]+vy[i_2j])/(24*DELTAX)
    value_dvx_dy = (27*vx[ij1]-27*vx[ij]-vx[ij2]+vx[ij_1]) / (24*DELTAY)

    # broadcast
    mem[3] = makevector(mem[3], ij,  adbroadcast(bx[1,2:end], mem[3][ij], 1) + adbroadcast(ax[1, 2:end], value_dvy_dx, 1))
    mem[4] = makevector(mem[4], ij,  adbroadcast(by[2,1:end-1], mem[4][ij], 2) + adbroadcast(ay[2, 1:end-1], value_dvx_dy, 2));

    value_dvy_dx = adbroadcast(value_dvy_dx, kx[1,2:end], 3) + mem[3][ij]
    value_dvx_dy = adbroadcast(value_dvx_dy, ky[1,1:end-1], 4) + mem[4][ij]

    σxy = scatter_add_op(σxy, ij, μ_*(value_dvy_dx + value_dvx_dy) * DELTAT )

    return vx, vy, σxx, σyy, σxy,mem
end


function fw3(param::ElasticPropagatorParams, ρ, λ, μ, vx, vy, σxx, σyy, σxy,mem,
    ax, bx, ay, by, kx, ky)
    NX, NY = param.NX, param.NY
    DELTAX, DELTAY, DELTAT = param.DELTAX, param.DELTAY, param.DELTAT
    i_2j = getid(param, 1:NX-1, 3:NY+1)
    i_1j = getid(param, 2:NX, 3:NY+1)
    ij = getid(param, 3:NX+1, 3:NY+1)
    i1j = getid(param, 4:NX+2, 3:NY+1)

    ij_2 = getid(param, 3:NX+1, 1:NY-1)
    ij_1 = getid(param,3:NX+1, 2:NY)
    ij1 = getid(param,3:NX+1, 4:NY+2)

    value_dsigmaxx_dx = (27*σxx[ij]-27*σxx[i_1j]-σxx[i1j]+σxx[i_2j]) / (24*DELTAX);
    value_dsigmaxy_dy = (27*σxy[ij]-27*σxy[ij_1]-σxy[ij1]+σxy[ij_2]) / (24*DELTAY);

    mem[5] = makevector(mem[5],ij, adbroadcast(bx[1,2:end], mem[5][ij], 1) + adbroadcast(ax[1,2:end], value_dsigmaxx_dx, 1))
    mem[6] = makevector(mem[6],ij, adbroadcast(by[1,2:end], mem[6][ij], 2) + adbroadcast(ay[1,2:end], value_dsigmaxy_dy, 2))

    value_dsigmaxx_dx = adbroadcast(value_dsigmaxx_dx, kx[1,2:end], 3) + mem[5][ij]
    value_dsigmaxy_dy = adbroadcast(value_dsigmaxy_dy, ky[1,2:end], 4) + mem[6][ij]

    vx = scatter_add_op(vx, ij, (value_dsigmaxx_dx + value_dsigmaxy_dy) * DELTAT / ρ[ij])
    return vx, vy, σxx, σyy, σxy, mem
end



function fw4(param::ElasticPropagatorParams, ρ, λ, μ, vx, vy, σxx, σyy, σxy,mem,
    ax, bx, ay, by, kx, ky)
    NX, NY = param.NX, param.NY
    DELTAX, DELTAY, DELTAT = param.DELTAX, param.DELTAY, param.DELTAT
    i_1j = getid(param, 1:NX-1, 2:NY)
    ij = getid(param, 2:NX, 2:NY)
    i1j = getid(param, 3:NX+1, 2:NY)
    i2j = getid(param, 4:NX+2, 2:NY)
    i1j1 = getid(param, 3:NX+1, 3:NY+1)

    ij_1 = getid(param, 2:NX, 1:NY-1)
    ij1 = getid(param, 2:NX, 3:NY+1)
    ij2 = getid(param, 2:NX, 4:NY+2)

    ρ_ = 0.25 * (ρ[ij] + ρ[i1j] + ρ[i1j1] + ρ[ij1]);

    value_dsigmaxy_dx = (27*σxy[i1j]-27*σxy[ij]-σxy[i2j]+σxy[i_1j]) / (24*DELTAX);
    value_dsigmayy_dy = (27*σyy[ij1]-27*σyy[ij]-σyy[ij2]+σyy[ij_1]) / (24*DELTAY);

    mem[7] = makevector(mem[7], ij, adbroadcast(bx[2,1:end-1], mem[7][ij], 1) + adbroadcast(ax[2,1:end-1] , value_dsigmaxy_dx, 1))
    mem[8] = makevector(mem[8], ij, adbroadcast(by[2,1:end-1], mem[8][ij], 2) + adbroadcast(ay[2,1:end-1] , value_dsigmayy_dy, 2))

    value_dsigmaxy_dx = adbroadcast(value_dsigmaxy_dx, kx[2,1:end-1], 3) + mem[7][ij]
    value_dsigmayy_dy = adbroadcast(value_dsigmayy_dy, ky[2,1:end-1], 4) + mem[8][ij]

    vy = scatter_add_op( vy, ij, (value_dsigmaxy_dx + value_dsigmayy_dy) * DELTAT / ρ_)
    return vx, vy, σxx, σyy, σxy,mem
end

function one_step(param::ElasticPropagatorParams, ρ, λ, μ, vx, vy, σxx, σyy, σxy,mem,
    ax, bx, ay, by, kx, ky, srci, srcj, srcv, srctype)
    add_source = load_op_and_grad("$(@__DIR__)/../deps/CustomOps/build/libADSeismic","add_source", multiple=true)
    vx, vy, σxx, σyy, σxy,mem = fw1(param, ρ, λ, μ, vx, vy, σxx, σyy, σxy,mem,
                    ax, bx, ay, by, kx, ky)
    vx, vy, σxx, σyy, σxy,mem = fw2(param, ρ, λ, μ, vx, vy, σxx, σyy, σxy,mem,
                    ax, bx, ay, by, kx, ky)
    vx, vy, σxx, σyy, σxy,mem = fw3(param, ρ, λ, μ, vx, vy, σxx, σyy, σxy,mem,
                    ax, bx, ay, by, kx, ky)
    vx, vy, σxx, σyy, σxy,mem = fw4(param, ρ, λ, μ, vx, vy, σxx, σyy, σxy,mem,
                    ax, bx, ay, by, kx, ky)    
    σxx, σyy, σxy,vx, vy  = add_source(σxx, σyy, σxy,vx,vy,srci,srcj,srctype,constant(param.NX),constant(param.NY),srcv)
    return σxx, σyy, σxy,vx, vy, mem
end




function compute_PML_Params(param::ElasticPropagatorParams)
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
    
    a_x = zeros(NX)
    a_y = zeros(NY)
    a_x_half = zeros(NX)
    a_y_half = zeros(NY)
    b_x = zeros(NX)
    b_y = zeros(NY)
    b_x_half = zeros(NX)
    b_y_half = zeros(NY)
    d_x = zeros(NX)
    d_y = zeros(NY)
    d_x_half = zeros(NX)
    d_y_half = zeros(NY)
    K_x = ones(NX)
    K_y = ones(NY)
    K_x_half = ones(NX)
    K_y_half = ones(NY)
    alpha_x = zeros(NX)
    alpha_y = zeros(NY)
    alpha_x_half = zeros(NX)
    alpha_y_half = zeros(NY)
    
    for i = 1:NX
        xval = DELTAX * (i-1)
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
    
    for j = 1:NY
        
        yval = DELTAY * (j-1)
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
    
    
    return ax,bx,kx,ay,by,ky
    
end


# # not in use any more
# function ElasticPropagator(src::ElasticSource, λ::Array{Float64},
#     ρ::Array{Float64}, μ::Array{Float64}, param::Union{ElasticPropagatorParams,Missing}=missing)
#     if ismissing(param)
#         param = ElasticPropagatorParams()
#     end
#     NSTEP = param.NSTEP
#     if size(src.srcv,1)<NSTEP
#         error("ADSeismic: length of sources=$(size(src.srci,1)) and NSTEP=$NSTEP does not match")
#     end
#     if size(src.srcv,1)!=NSTEP
#         @warn("ADSeismic: length of sources and NSTEP does not match $(size(src.srci,1))!=$NSTEP")
#     end
#     vx_arr = TensorArray(NSTEP+1); vx_arr = write(vx_arr, 1, zeros(param.NX+2, param.NY+2))
#     vy_arr = TensorArray(NSTEP+1); vy_arr = write(vy_arr, 1, zeros(param.NX+2, param.NY+2))
#     sigmaxx_arr = TensorArray(NSTEP+1); sigmaxx_arr = write(sigmaxx_arr, 1, zeros(param.NX+2, param.NY+2))
#     sigmayy_arr = TensorArray(NSTEP+1); sigmayy_arr = write(sigmayy_arr, 1, zeros(param.NX+2, param.NY+2))
#     sigmaxy_arr = TensorArray(NSTEP+1); sigmaxy_arr = write(sigmaxy_arr, 1, zeros(param.NX+2, param.NY+2))
#     mem_arr = TensorArray(NSTEP+1); mem_arr = write(mem_arr, 1, zeros(8, param.NX+2, param.NY+2))
    
    
#     ax,bx,kx,ay,by,ky = compute_PML_Params(param)
    
    
#     function condition(i, vx_arr, vy_arr, sigmaxx_arr, sigmayy_arr, sigmaxy_arr, mem_arr)
#         i<=NSTEP+1
#     end
    
#     function body(i, vx_arr, vy_arr, sigmaxx_arr, sigmayy_arr, sigmaxy_arr, mem_arr)
#         sigmaxx,sigmayy,sigmaxy,vx,vy,mem = 
#             read(sigmaxx_arr, i-1), read(sigmayy_arr, i-1), read(sigmaxy_arr, i-1), 
#             read(vx_arr, i-1), read(vy_arr, i-1), read(mem_arr, i-1)
#         srci,srcj,srctype,srcv = ElasticSourceAtTimeT(src, i-1) 
#         # @show srci, srcj, srctype, srcv
#         # error()
#         sigmaxx, sigmayy, sigmaxy, vx,vy, mem = elastic_wave_op(λ,μ,ρ,sigmaxx,sigmayy,sigmaxy,vx,vy,
#                 mem,constant(ax),constant(bx),constant(kx),constant(zeros(length(ax))),constant(ay),constant(by),constant(ky),constant(zeros(length(bx))),constant(param.DELTAX),constant(param.DELTAY),constant(param.DELTAT),
#                 srci,srcj,srctype,srcv)
#         sigmaxx_arr,sigmayy_arr,sigmaxy_arr,vx_arr,vy_arr,mem_arr = 
#             write(sigmaxx_arr, i, sigmaxx), write(sigmayy_arr, i, sigmayy), write(sigmaxy_arr, i, sigmaxy),
#                     write(vx_arr, i, vx), write(vy_arr, i, vy), write(mem_arr, i, mem)
#         op = tf.cond(tf.equal(tf.math.floormod(i,param.IT_DISPLAY),1), ()->tf.print("time step = ", i-1, " out of $NSTEP"), ()->tf.no_op())
#         i = bind(i, op)
#         i+1, vx_arr, vy_arr, sigmaxx_arr, sigmayy_arr, sigmaxy_arr, mem_arr
#     end
#     i = constant(2, dtype=Int32)
#     _, vx_arr, vy_arr, sigmaxx_arr, sigmayy_arr, sigmaxy_arr, mem_arr = 
#         while_loop(condition, body, [i, vx_arr, vy_arr, sigmaxx_arr, sigmayy_arr, sigmaxy_arr, mem_arr])
#     ElasticPropagator(
#             param,
#             src,
#             vx_arr,
#             vy_arr,
#             sigmaxx_arr,
#             sigmayy_arr,
#             sigmaxy_arr,
#             mem_arr
#     )
# end


################################################ ACOUSTIC ################################################################## 

# todo:
# one_step(::AcousticPropagatorParams, ::PyCall.PyObject, ::PyCall.PyObject, ::PyCall.PyObject, ::PyCall.PyObject, ::PyCall.PyObject, ::PyCall.PyObject, ::PyCall.PyObject)
function acoustic_one_step_customop(param::AcousticPropagatorParams, w::PyObject, wold::PyObject, φ, ψ, σ::PyObject, τ::PyObject, c::PyObject)
    phi = φ
    psi = ψ
    sigma = σ
    tau = τ
    dt = param.DELTAT
    hx, hy = param.DELTAX, param.DELTAY
    nx, ny = param.NX, param.NY
    acoustic_one_step_ = load_op_and_grad("$(@__DIR__)/../deps/CustomOps/build/libADSeismic", "acoustic_one_step", multiple=true)
    w,wold,phi,psi,sigma,tau,c,dt,hx,hy,nx,ny = convert_to_tensor(Any[w,wold,phi,psi,sigma,tau,c,dt,hx,hy,nx,ny], [Float64,Float64,Float64,Float64,Float64,Float64,Float64,Float64,Float64,Float64,Int64,Int64])
    u, φ, ψ = acoustic_one_step_(w,wold,phi,psi,sigma,tau,c,dt,hx,hy,nx,ny)
    set_shape(u, (param.NX+2)*(param.NY+2)), set_shape(φ, (param.NX+2)*(param.NY+2)), set_shape(ψ, (param.NX+2)*(param.NY+2))
end


function acoustic_one_step_customop_cpu(param::AcousticPropagatorParams, w::PyObject, wold::PyObject, φ, ψ, σ::PyObject, τ::PyObject, c::PyObject)
    phi = φ
    psi = ψ
    sigma = σ
    tau = τ
    dt = param.DELTAT
    hx, hy = param.DELTAX, param.DELTAY
    nx, ny = param.NX, param.NY
    acoustic_one_step_ = load_op_and_grad("$(@__DIR__)/../deps/CustomOps/build/libADSeismic", "acoustic_one_step_cpu", multiple=true)
    w,wold,phi,psi,sigma,tau,c,dt,hx,hy,nx,ny = convert_to_tensor(Any[w,wold,phi,psi,sigma,tau,c,dt,hx,hy,nx,ny], [Float64,Float64,Float64,Float64,Float64,Float64,Float64,Float64,Float64,Float64,Int64,Int64])
    u, φ, ψ = acoustic_one_step_(w,wold,phi,psi,sigma,tau,c,dt,hx,hy,nx,ny)
    set_shape(u, (param.NX+2)*(param.NY+2)), set_shape(φ, (param.NX+2)*(param.NY+2)), set_shape(ψ, (param.NX+2)*(param.NY+2))
end

function acoustic_one_step_customop_ref(param::AcousticPropagatorParams, w::PyObject, wold::PyObject, φ, ψ, σ::PyObject, τ::PyObject, c::PyObject)
    Δt = param.DELTAT
    hx, hy = param.DELTAX, param.DELTAY
    IJ, IpJ, InJ, IJp, IJn, IpJp, IpJn, InJp, InJn =
        param.IJ, param.IpJ, param.InJ, param.IJp, param.IJn, param.IpJp, param.IpJn, param.InJp, param.InJn
        
    u = (2 - σ[IJ]*τ[IJ]*Δt^2 - 2*Δt^2/hx^2 * c[IJ] - 2*Δt^2/hy^2 * c[IJ]) * w[IJ] +
            c[IJ] * (Δt/hx)^2  *  (w[IpJ]+w[InJ]) +
            c[IJ] * (Δt/hy)^2  *  (w[IJp]+w[IJn]) +
            (Δt^2/(2hx))*(φ[IpJ]-φ[InJ]) +
            (Δt^2/(2hy))*(ψ[IJp]-ψ[IJn]) -
            (1 - (σ[IJ]+τ[IJ])*Δt/2) * wold[IJ] 
    u = u / (1 + (σ[IJ]+τ[IJ])/2*Δt)
    u = scatter_nd_ops(IJ, u, (param.NX+2)*(param.NY+2))
    φ = (1. -Δt*σ[IJ]) * φ[IJ] + Δt * c[IJ] * (τ[IJ] -σ[IJ])/2hx *  
        (w[IpJ]-w[InJ])
    ψ = (1. -Δt*τ[IJ]) * ψ[IJ] + Δt * c[IJ] * (σ[IJ] -τ[IJ])/2hy * 
        (w[IJp]-w[IJn])
    φ = scatter_nd_ops(IJ, φ, (param.NX+2)*(param.NY+2))
    ψ = scatter_nd_ops(IJ, ψ, (param.NX+2)*(param.NY+2))
    u, φ, ψ
end


function one_step(param::AcousticPropagatorParams, w::PyObject, wold::PyObject, φ, ψ, σ::PyObject, τ::PyObject, c::PyObject)
    Δt = param.DELTAT
    hx, hy = param.DELTAX, param.DELTAY
    IJ, IpJ, InJ, IJp, IJn, IpJp, IpJn, InJp, InJn =
        param.IJ, param.IpJ, param.InJ, param.IJp, param.IJn, param.IpJp, param.IpJn, param.InJp, param.InJn
        
    u = (2 - σ[IJ]*τ[IJ]*Δt^2 - 2*Δt^2/hx^2 * c[IJ] - 2*Δt^2/hy^2 * c[IJ]) * w[IJ] +
            c[IJ] * (Δt/hx)^2  *  (w[IpJ]+w[InJ]) +
            c[IJ] * (Δt/hy)^2  *  (w[IJp]+w[IJn]) +
            (Δt^2/(2hx))*(φ[IpJ]-φ[InJ]) +
            (Δt^2/(2hy))*(ψ[IJp]-ψ[IJn]) -
            (1 - (σ[IJ]+τ[IJ])*Δt/2) * wold[IJ] 
    u = u / (1 + (σ[IJ]+τ[IJ])/2*Δt)
    u = scatter_nd_ops(IJ, u, (param.NX+2)*(param.NY+2))
    φ = (1. -Δt*σ[IJ]) * φ[IJ] + Δt * c[IJ] * (τ[IJ] -σ[IJ])/2hx *  
        (u[IpJ]-u[InJ])
    ψ = (1. -Δt*τ[IJ]) * ψ[IJ] + Δt * c[IJ] * (σ[IJ] -τ[IJ])/2hy * 
        (u[IJp]-u[IJn])
    φ = scatter_nd_ops(IJ, φ, (param.NX+2)*(param.NY+2))
    ψ = scatter_nd_ops(IJ, ψ, (param.NX+2)*(param.NY+2))
    u, φ, ψ
end

"""
AcousticPropagatorSolver(param::AcousticPropagatorParams, src::AcousticSource, c::Union{PyObject, Array{Float64, 2}})

# Arguments
- 'param::AcousticPropagatorParams': model configuration parameters, such as NX, NT
- 'src::AcousticSource': source locations
- 'c::Union{PyObject, Array{Float64, 2}}': velocity model

# Return
- 'AcousticPropagator': forward calculation results, like displacement u
"""
function AcousticPropagatorSolver(param::AcousticPropagatorParams, src::AcousticSource, c::Union{PyObject, Array{Float64, 2}})

    c = tf.reshape(convert_to_tensor(c), (-1,))^2 ## convert to c^2
    compute_PML_Params!(param)

    σij = constant(param.Σx'[:])
    τij = constant(param.Σy'[:])

    tu = TensorArray(param.NSTEP+1; clear_after_read=false)
    tφ = TensorArray(param.NSTEP+1; clear_after_read=true)
    tψ = TensorArray(param.NSTEP+1; clear_after_read=true)

    function condition(i, ta, tφ, tψ)
        i<=param.NSTEP+1
    end

    function body(i, ta, tφ, tψ)
        
        if param.PropagatorKernel==0
            one_step_ = one_step
        elseif param.PropagatorKernel==1 && HAS_GPU
            @info "Use GPU custom one step..."
            one_step_ = acoustic_one_step_customop
        elseif param.PropagatorKernel==1 && !HAS_GPU
            @info "Use CPU custom one step..."
            one_step_ = acoustic_one_step_customop_cpu
        elseif param.PropagatorKernel==2
            @info "Use reference one step..."
            one_step_ = acoustic_one_step_customop_ref
        end
        u, φ, ψ = one_step_(param, read(ta, i-1), read(ta, i-2), read(tφ, i-1), read(tψ, i-1), σij, τij, c)
        srci, srcj, srcv = AcousticSourceAtTimeT(src, i-1)

        if param.IT_DISPLAY>0
            op = tf.cond(tf.equal(tf.math.floormod(i,param.IT_DISPLAY),1), ()->tf.print("time step = ", i-1, " out of $(param.NSTEP)"), ()->tf.no_op())
            i = bind(i, op)
        end

        src_index = (srci - 1) * (param.NY+2) + srcj 
        u = scatter_add_op(u, src_index, srcv*param.DELTAT^2)

        ta_, tφ_, tψ_ = write(ta, i, u), write(tφ, i, φ), write(tψ, i, ψ)
        i+1, ta_, tφ_, tψ_
    end

    tu = write(tu, 1, constant(zeros((param.NX+2)*(param.NY+2))))
    tφ = write(tφ, 1, constant(zeros((param.NX+2)*(param.NY+2))))
    tψ = write(tψ, 1, constant(zeros((param.NX+2)*(param.NY+2))))
    tu = write(tu, 2, constant(zeros((param.NX+2)*(param.NY+2))))
    tφ = write(tφ, 2, constant(zeros((param.NX+2)*(param.NY+2))))
    tψ = write(tψ, 2, constant(zeros((param.NX+2)*(param.NY+2))))
    i = constant(3, dtype=Int32)
    _, tu, tφ, tψ = while_loop(condition, body, [i,tu,tφ,tψ]; swap_memory=true)
    tu = tf.reshape(stack(tu), (param.NSTEP+1, param.NX+2, param.NY+2))
    tφ = tf.reshape(stack(tφ), (param.NSTEP+1, param.NX+2, param.NY+2))
    tψ = tf.reshape(stack(tψ), (param.NSTEP+1, param.NX+2, param.NY+2))

    AcousticPropagator(param, src, tu, tφ, tψ)
end

function compute_PML_Params!(param::AcousticPropagatorParams)
    NX, NY = param.NX, param.NY
    # computing damping coefficient
    c, R = param.vp_ref, param.Rcoef
    Lx = param.NPOINTS_PML * param.DELTAX
    Ly = param.NPOINTS_PML * param.DELTAY
    param.damping_x = c/Lx*log(1/R)
    param.damping_y = c/Ly*log(1/R)
    # @show c, Lx, log(1/R), param.damping_x, param.damping_y


    X = (0:param.NX+1)*param.DELTAX
    Y = (0:param.NY+1)*param.DELTAY
    Σx = zeros(param.NX+2, param.NY+2)
    Σy = zeros(param.NX+2, param.NY+2)
    for i = 1:param.NX+2
        for j = 1:param.NY+2
            Σx[i,j], Σy[i,j] = pml_helper(X[i], Y[j], param)
        end
    end

    param.Σx = Σx
    param.Σy = Σy
    param.IJ = getid2(2:NX+1, 2:NY+1, NX, NY)
    param.IJn = getid2(2:NX+1, 1:NY, NX, NY)
    param.IJp = getid2(2:NX+1, 3:NY+2, NX, NY)
    param.IpJ = getid2(3:NX+2, 2:NY+1, NX, NY)
    param.IpJp = getid2(3:NX+2, 3:NY+2, NX, NY)
    param.IpJn = getid2(3:NX+2, 1:NY, NX, NY)
    param.InJ = getid2(1:NX, 2:NY+1, NX, NY)
    param.InJn = getid2(1:NX, 1:NY, NX, NY)
    param.InJp = getid2(1:NX, 3:NY+2, NX, NY)
    return param
end


# short functions #
function ElasticReceiver(rcvi::Union{Array{Int64, 1},PyObject}, rcvj::Union{Array{Int64, 1},PyObject}, 
    rcvtype::Union{Array{Int64, 1},PyObject})
    ElasticReceiver(rcvi, rcvj, rcvtype, missing)
end

function ElasticSourceAtTimeT(src::ElasticSource,i::PyObject)
    convert_to_tensor(src.srci), convert_to_tensor(src.srcj), 
        convert_to_tensor(src.srctype), convert_to_tensor(src.srcv)[i]
end

function getid(param, a, b)
    idx = Int64[]
    for i = 1:length(a)
        for j = 1:length(b)
            push!(idx, (a[i]-1)*(param.NY+2)+b[j])
        end
    end
    idx
end

function adbroadcast(a, b, idx=1)        
    if idx==2
        n = div(length(b), length(a))
        repeat(a, n)*b
    elseif idx==1
        n = div(length(b), length(a))
        repeat(a', n)[:]*b
    elseif idx==4
        n = div(length(a), length(b))
        a/repeat(b, n)
        return a
    elseif idx==3
        n = div(length(a), length(b))
        a/repeat(b', n)[:]
        return a
    end
end

function makevector(m::PyObject, ij::Array{Int64}, o::PyObject)
    scatter_nd_ops(ij, o, length(m))
end

function SimulatedObservation!(ep::ElasticPropagator, rcv::ElasticReceiver)
    N = (ep.param.NX+2)*(ep.param.NY+2)
    get_receive = load_op_and_grad("$(@__DIR__)/../deps/CustomOps/build/libADSeismic", "get_receive")
    u = get_receive(tf.reshape(ep.vx, (ep.param.NSTEP+1, N)), 
                    tf.reshape(ep.vy, (ep.param.NSTEP+1, N)), 
                    tf.reshape(ep.sigmaxx, (ep.param.NSTEP+1, N)), 
                    tf.reshape(ep.sigmayy, (ep.param.NSTEP+1, N)), 
                    tf.reshape(ep.sigmaxy, (ep.param.NSTEP+1, N)),
                    convert_to_tensor(rcv.rcvi), convert_to_tensor(rcv.rcvj), convert_to_tensor(rcv.rcvtype),
                        constant(ep.param.NX), constant(ep.param.NY))
    rcv.rcvv = u
end



function AcousticSourceAtTimeT(src::AcousticSource,i::PyObject)
    convert_to_tensor(src.srci, dtype=Int64), convert_to_tensor(src.srcj, dtype=Int64), convert_to_tensor(src.srcv)[i]
end


"""
    SimulatedObservation!(ap::AcousticPropagator, rcv::AcousticReceiver)

Extract and save simulated displacement u into rcv::AcousticReceiver.
"""
function SimulatedObservation!(ap::AcousticPropagator, rcv::AcousticReceiver)
    idx = (rcv.rcvi .- 1) *  (ap.param.NY+2) + rcv.rcvj 
    u = reshape(ap.u, ap.param.NSTEP+1, (ap.param.NX+2)*(ap.param.NY+2))[:, idx]
    rcv.rcvv = u
end


"""
    pml_helper(x::Float64, y::Float64, param::AcousticPropagatorParams)

Computing the PML profile. 
"""
function pml_helper(x::Float64, y::Float64, param::AcousticPropagatorParams)
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

function getid2(a, b, NX, NY)
    idx = Int64[]
    for i = 1:length(a)
        for j = 1:length(b)
            push!(idx, (a[i]-1)*(NY+2)+b[j])
        end
    end
    return idx
end