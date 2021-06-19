export visualize_wavefield, plot_result, add_initial_model,
    Ricker, Gauss, compute_lame_parameters, 
    gradtest, compute_loss_and_grads_GPU, compute_forward_GPU, variable_source, 
    sampling_compute_loss_and_grads_GPU, 
    sampling_compute_loss_and_grads_GPU_v2,
    sampling_compute_loss_and_grads_GPU_v3,
    visualize_model, aggregate_wavefield

function visualize_model(vp::Array{Float64, 2}, params::Union{ElasticPropagatorParams,AcousticPropagatorParams, MPIAcousticPropagatorParams})
    
    clf()
    if isa(params, MPIAcousticPropagatorParams)
        pcolormesh([0:params.NX-1;]*params.DELTAX/1e3,[0:params.NY-1;]*params.DELTAY/1e3,  Array(vp'))
    else    
        pcolormesh([0:params.NX+1;]*params.DELTAX/1e3,[0:params.NY+1;]*params.DELTAY/1e3,  Array(vp'))
    end
    axis("scaled")
    colorbar(shrink=0.4)
    xlabel("x (km)")
    ylabel("z (km)")
    gca().invert_yaxis()
    tight_layout()
    
end

function aggregate_wavefield(param::Union{MPIAcousticPropagatorParams, MPIElasticPropagatorParams}, vals::Array{Array{Float64, 3}, 1})
    M, N = param.M, param.N 
    nt, m, n = size(vals[1])
    Vals = zeros(nt, m*M, n*N)
    k = 0
    for i = 1:M 
        for j = 1:N 
            k += 1
            Vals[:, (i-1)*m+1:i*m, (j-1)*n+1:j*n] = vals[k]
        end
    end
    Vals
end

function visualize_wavefield(val::Array{Array{Float64, 3}, 1}, 
    param::Union{MPIAcousticPropagatorParams, MPIElasticPropagatorParams})
    val = aggregate_wavefield(param, val)
    visualize_wavefield(val, param)
end

"""
    visualize_wavefield(val::Array{Float64, 3}, 
        param::Union{ElasticPropagatorParams,AcousticPropagatorParams})
    visualize_wavefield(val::Array{Array{Float64, 3}, 1}, 
        param::Union{MPIAcousticPropagatorParams, MPIElasticPropagatorParams})

Visualizes the wavefield and returns the handle of the animation. You can save the animation to gif via 

```julia
p = visualize_wavefield(...)
saveanim(p, "myfigure.gif")
```
"""
function visualize_wavefield(val::Array{Float64, 3}, 
    param::Union{ElasticPropagatorParams,AcousticPropagatorParams, MPIAcousticPropagatorParams, MPIElasticPropagatorParams})
    
    figure()
    vmin = -3std(val)
    vmax = 3std(val)
    dt = max(1, div(param.NSTEP, 20))

    pl = matshow(val[1,:,:]', cmap="jet", vmin=vmin, vmax=vmax)
    t = title("Time = $(round(0.0, digits=2))")
    colorbar(shrink=0.8)
    gca().xaxis.set_ticks_position("bottom")
    xlabel("X")
    ylabel("Y")

    function update(i)
        pl[:set_data](val[i,:,:]')
        t.set_text("Time = $(round((i-1)*param.DELTAT, digits=3))")
    end
    p = animate(update, [2:dt:size(val, 1);])

end
    

function plot_result(sess, var, feed_dict, iter; figure_dir::String="figures", result_dir::String="results", var_name=nothing)

    if !isdir(figure_dir)
        figure_dir="./"
    end
    if !isdir(result_dir)
        result_dir=figure_dir
    end

    x = run(sess, var, feed_dict=feed_dict)

    std_images = zeros(size(x,2), size(x,3)) * 0
    mean_images = zeros(size(x,2), size(x,3)) * 0
    for i = 1:size(x,2)
        for j = 1:size(x, 3)
            std_images[i,j] = std(x[:, i, j, 1])
            mean_images[i,j] = mean(x[:, i, j, 1])
        end
    end

    figure()
    imshow(std_images')
    colorbar(shrink=0.5)
    title("Iteration = $iter")
    if isnothing(var_name)
        savefig(joinpath(figure_dir, "std_$(lpad(iter,5,"0")).png"), bbox_inches="tight")
        writedlm(joinpath(result_dir, "std_$(lpad(iter,5,"0")).txt"), std_images')
    else
        savefig(joinpath(figure_dir, "std_$(var_name)_$(lpad(iter,5,"0")).png"), bbox_inches="tight")
        writedlm(joinpath(result_dir, "std_$(var_name)_$(lpad(iter,5,"0")).txt"), std_images')
    end

    figure()
    fig, ax = subplots(div(size(x)[1],2), 2)
    for i = 1:div(size(x)[1],2)
        for j = 1:2
            ax[i, j].get_xaxis().set_visible(false)
            ax[i, j].get_yaxis().set_visible(false)
            k = (i-1) * 2 + j
            pcm= ax[i, j].imshow(x[k,:,:]')
            # fig.colorbar(pcm, ax=ax[i,j])
        end
    end
    subplots_adjust(wspace=0, hspace=0)
    suptitle("Iteration = $iter")
    if isnothing(var_name)
        savefig(joinpath(figure_dir, "inv_$(lpad(iter,5,"0")).png"), bbox_inches="tight")
        writedlm(joinpath(result_dir, "inv_$(lpad(iter,5,"0")).txt"), mean_images')
    else
        savefig(joinpath(figure_dir, "inv_$(var_name)_$(lpad(iter,5,"0")).png"), bbox_inches="tight")
        writedlm(joinpath(result_dir, "inv_$(var_name)_$(lpad(iter,5,"0")).txt"), mean_images')
    end
    # writedlm(joinpath(result_dir, "inv_$(lpad(iter,5,"0")).txt"), vp*scale)
    close("all")
end


function add_initial_model(x, v)
    if ndims(x) == 3
        size_x = size(x)[2:3]
    else
        size_x = size(x)
    end
    if size_x <= size(v)
        i = (size(v)[1]-size_x[1])÷2 +1 :(size(v)[1]-size_x[1])÷2 + size_x[1]
        j = (size(v)[2]-size_x[2])÷2 +1 : (size(v)[2]-size_x[2])÷2 + size_x[2]
        if ndims(x) == 3
            v = stack([scatter_add(v, i, j, x[k]) for k in 1:size(x)[1]])
        else
            v = scatter_add(v, i, j, x)
        end
        @warn "size(x) <= size(vp)", size_x, size(v)
    elseif  size_x > size(v)
        if ndims(x) == 3
            v = v + tf.slice(x, (size(x).-(size(x)[1], size(v)...)).÷2, (size(x)[1], size(v)...))
        else
            v = v + tf.slice(x, (size(x).-size(v)).÷2, size(v))
        end
        @warn "size(x) > size(vp)", size_x, size(v)
    else
        error("Size error: ", size(v), size_x)
    end
    return v
end

"""
    Ricker(epp::Union{ElasticPropagatorParams, AcousticPropagatorParams}, 
    a::Union{PyObject, <:Real}, 
    shift::Union{PyObject, <:Real}, 
    amp::Union{PyObject, <:Real}=1.0)

Returns a Ricker wavelet (a tensor). 
- `epp`: a `ElasticPropagatorParams` or an `AcousticPropagatorParams`
- `a`: Width parameter
- `shift`: Center of the Ricker wavelet
- `amp`: Amplitude of the Ricker wavelet

```math
f(x) = \\mathrm{amp}A (1 - x^2/a^2) exp(-x^2/2 a^2)
```
where 
```math
A = 2/sqrt(3a)pi^1/4
```
"""
function Ricker(epp::Union{ElasticPropagatorParams, AcousticPropagatorParams, MPIAcousticPropagatorParams, MPIElasticPropagatorParams}, 
        a::Union{PyObject, <:Real}, 
        shift::Union{PyObject, <:Real}, 
        amp::Union{PyObject, <:Real}=1.0)
    a = convert_to_tensor(a)
    shift = convert_to_tensor(shift)
    amp = convert_to_tensor(amp)
    NT, T = epp.NSTEP, epp.NSTEP*epp.DELTAT
    A = 2 / (sqrt(3 * a) * (pi^0.25))
    wsq = a^2
    vec = collect(1:NT)-shift
    xsq = vec^2
    mod = (1 - xsq / wsq)
    gauss = exp(-xsq / (2 * wsq))
    total = amp * A * mod * gauss
    return total
end

function Gauss(epp::Union{ElasticPropagatorParams, AcousticPropagatorParams, MPIAcousticPropagatorParams, MPIElasticPropagatorParams},
        a::Union{<:Real, PyObject}, 
        shift::Union{<:Real, PyObject, Missing} = missing,
        amp::Union{<:Real, PyObject} = 1.0)
    if ismissing(shift)
        shift = 1.2/a
    end
    amp = convert_to_tensor(amp)
    a = convert_to_tensor(a)
    shift = convert_to_tensor(shift, dtype=Float64)
    t = collect(0:epp.NSTEP-1)*epp.DELTAT
    A = π^2*a^2
    # -a*2.0*a*(t - t0) .* exp(-a*(t - t0)^2)
    amp * 2.0 * A * exp(-A*(t - shift)^2)
end


function compute_lame_parameters(NX::Int, NY::Int, vp::Float64, vs::Float64, rho::Float64)
    λ = zeros(NX+2, NY+2)
    μ = zeros(NX+2, NY+2)
    ρ = zeros(NX+2, NY+2)
    for j = 1:NY+2
        for i = 1:NX+2
            ρ[i,j] = rho
            μ[i,j] = rho*vs*vs
            λ[i,j] = rho*(vp*vp - 2. * vs*vs)
        end
    end
    λ, μ, ρ
end

function compute_lame_parameters(vp::Union{PyObject, Array{Float64,2}}, vs::Union{PyObject, Array{Float64,2}}, 
                                 rho::Union{PyObject, Array{Float64,2}})
    ρ = rho
    μ = rho .* vs .* vs
    λ = rho .* (vp .* vp - 2. * vs .* vs)
    return λ, μ, ρ
end

function compute_lame_parameters(param::MPIElasticPropagatorParams, vp::Float64, vs::Float64, rho::Float64)
    compute_lame_parameters(param.n+2, param.n+2, vp, vs, rho)
end

function compute_properties(vp::Union{PyObject, Array{Float64}}, 
                            vs::Union{PyObject, Array{Float64}}, 
                            ρ::Union{PyObject, Array{Float64}})
    vs = convert_to_tensor(vs)
    vp = convert_to_tensor(vp)
    ρ = convert_to_tensor(ρ)
    μ = ρ .* vs .* vs
    λ = ρ .* vp .* vp - 2 * μ
    λ, μ, ρ
end

function gradtest(sess::PyObject, o::PyObject, g::PyObject, loss::PyObject)
    sess = Session(); init(sess)
    X, G, L = run(sess, [o, g, loss])
    v_ = rand(size(X)...)*median(X)

    gs_ =  @. 1 / 10^(1:5)
    s = zeros(5)
    w = zeros(5)
    for i = 1:5
        @info i
        g_ = gs_[i]
        run(sess, assign(o, X + g_*v_))
        Li = run(sess, loss)
        s[i] = Li - L
        w[i] = Li - L - g_ * sum(v_ .* G)
    end

    close("all")
    loglog(gs_, abs.(s), "*-", label="finite difference")
    loglog(gs_, abs.(w), "+-", label="automatic differentiation")
    loglog(gs_, gs_.^2 * 0.5*abs(w[1])/gs_[1]^2, "--",label="\$\\mathcal{O}(\\gamma^2)\$")
    loglog(gs_, gs_ * 0.5*abs(s[1])/gs_[1], "--",label="\$\\mathcal{O}(\\gamma)\$")

    plt.gca().invert_xaxis()
    legend()
    xlabel("\$\\gamma\$")
    ylabel("Error")
end

"""
    compute_loss_and_grads_GPU(model::Function, src::Union{Array{AcousticSource},Array{ElasticSource}},
    rcv_sim::Union{Array{AcousticReceiver}, Array{ElasticReceiver}}, Rs::Array{Array{Float64}}, vs::Union{PyObject, Array{PyObject}})

Computes the loss and the gradients of the model on all available GPUs. 
- `model`, `src`, `rcv_sim`: Model, Source and Receiver returned by [`load_acoustic_source`](@ref), [`load_acoustic_model`](@ref), 
    [`load_acoustic_receiver`](@ref) or [`load_elastic_source`](@ref), [`load_elastic_model`](@ref), 
    [`load_elastic_receiver`](@ref)
- `Rs`: True receiver data
- `vs`: Variables to be optimized. It or its elements must have `ref` type.
"""
function compute_loss_and_grads_GPU(model::Function, src::Union{Array{AcousticSource},Array{ElasticSource}},
         rcv_sim::Union{Array{AcousticReceiver}, Array{ElasticReceiver}}, Rs::Array{Array{Float64,2},1}, vs::Union{PyObject, Array{PyObject}}; reg::Float64=0.0)

    flag = isa(vs, PyObject)
    function run_on_gpu_device(gpu_id, jobs)
        local loss, g
        @pywith tf.device("/gpu:$(gpu_id)") begin
            [SimulatedObservation!(model(src[i]), rcv_sim[i]) for i = jobs]
            loss = sum([sum((rcv_sim[i].rcvv-Rs[i])^2) for i = jobs]) + reg * tf.cast(sum(sum.(vs.^2)), tf.float64)
            g = gradients(loss, vs)
        end
        return loss, g
    end

    n = length(Rs)
    n_gpu = min(length(use_gpu()), n)
    if n_gpu==0
        error("No GPU resources available")
    end

    losses = Array{PyObject}(undef, n_gpu)
    gs = Array{flag ? PyObject : Array{PyObject}}(undef, n_gpu)
    for i = 1:n_gpu
        jobs = [k for k=1:n if k%n_gpu==i-1]
        println("GPU $i --> sources $jobs")
        losses[i], gs[i] = run_on_gpu_device(i-1, jobs)
    end
    gs = [gradients_GPU(x) for x in losses]
    return losses, gs
end

# function sampling_compute_loss_and_grads_GPU(models, src::AcousticSource,
#     rcv_sim::Array{AcousticReceiver}, 
#     Rs::Array{PyObject}, reg::Float64=1.0)
#     local loss, d
#     @show length(models), length(rcv_sim)
#     @assert length(models)==length(rcv_sim)
#     n_models = length(models)
#     for model in models
#         if model.param.IT_DISPLAY!=0
#             @warn "To use GPU, IT_DISPLAY must be 0. Setting IT_DISPLAY to 0..."
#             model.param.IT_DISPLAY = 0
#         end
#     end

#     function run_on_gpu_device!(gpu_id, jobs)
#         local d
#         @pywith tf.device("/gpu:$(gpu_id)") begin
#             [SimulatedObservation!(models[i](src), rcv_sim[i]) for i = jobs]
#             x = vcat([reshape(rcv_sim[i].rcvv,1,-1) for i in jobs]...)
#             y = vcat([reshape(r, 1, -1) for r in Rs]...)
#             d = dist(x,y,1)
#             @info size(x), size(y), size(d)
#         end
#         return d
#     end

#     n_gpu = min(length(use_gpu()), n_models)

#     if n_gpu==0
#         @warn "not using GPU"
#         [SimulatedObservation!(models[i](src), rcv_sim[i]) for i = 1:n_models]
#     else
#         ds = Array{PyObject}(undef, n_gpu)
#         for i = 1:n_gpu
#             jobs = [k for k=1:n_models if k%n_gpu==i-1]
#             println("GPU $i --> sources $jobs")
#             ds[i] = run_on_gpu_device!(i-1, jobs)
#         end
#     end

#     @pywith tf.device("/cpu:0") begin
#         # x = vcat([reshape(rcv_sim[i].rcvv,1,-1) for i = 1:n_models]...)
#         # y = vcat([reshape(r, 1, -1) for r in Rs]...)
#         # @info size(x), size(y)
#         # d = dist(x,y,1)
#         d = vcat(ds...)
#         a = tf.ones((size(d,1),), dtype=tf.float64)/Float64(size(d,1))
#         b = tf.ones((size(d,2),), dtype=tf.float64)/Float64(size(d,2))
#         loss = sinkhorn(a, b, d, reg=reg)
#     end
#     return loss, d
# end

function sampling_compute_loss_and_grads_GPU(models, src::AcousticSource, rcv_sim::Array{AcousticReceiver}, Rs::Array{PyObject}, reg::Float64=1.0)
    
    local loss, d
    @show length(models), length(rcv_sim)
    @assert length(models)==length(rcv_sim)
    n_models = length(models)

    function run_on_gpu_device!(gpu_id, jobs)
        local d
        @pywith tf.device("/gpu:$(gpu_id)") begin
            [SimulatedObservation!(models[i](src), rcv_sim[i]) for i = jobs]
            x = vcat([reshape(rcv_sim[i].rcvv,1,-1) for i in jobs]...)
            y = vcat([reshape(r, 1, -1) for r in Rs]...)
            d = dist(x,y,1)
            # @info size(x), size(y), size(d)
        end
        return d
    end

    n_gpu = min(length(use_gpu()), n_models)

    if n_gpu==0
        @warn "not using GPU"
        [SimulatedObservation!(models[i](src), rcv_sim[i]) for i = 1:n_models]
    else
        ds = Array{PyObject}(undef, n_gpu)
        for i = 1:n_gpu
            jobs = [k for k=1:n_models if k%n_gpu==i-1]
            println("GPU $i --> sources $jobs")
            ds[i] = run_on_gpu_device!(i-1, jobs)
        end
    end

    @pywith tf.device("/cpu:0") begin
        d = vcat(ds...)
        a = tf.ones((size(d,1),), dtype=tf.float64)/Float64(size(d,1))
        b = tf.ones((size(d,2),), dtype=tf.float64)/Float64(size(d,2))
        loss = sinkhorn(a, b, d, reg=reg, method="lp")
    end
    return loss, d
end


"""
- `models` : nsrc * batch_size
- `src` : nsrc
- `rcv_sim` : nsrc * batch_size
- `Rs` : nsrc * sample_size
"""
function sampling_compute_loss_and_grads_GPU_v2(models, src, rcv_sim,  Rs; reg::Float64=1.0, method::String="sinkhorn")
    
    local loss
    nsrc, batch_size = size(models)
    @assert nsrc == length(src)
    @assert size(rcv_sim,1) == nsrc
    @assert size(rcv_sim,2) == batch_size
    @assert size(Rs,1)==nsrc
    sample_size = size(Rs,2)

    function run_on_gpu_device!(gpu_id, src_ids)
        local ds
        @pywith tf.device("/gpu:$(gpu_id)") begin
            x = Any[[] for i=1:length(src_ids)]
            y = Any[[] for i=1:length(src_ids)]
            for (k,i) in enumerate(src_ids)
                for j = 1:batch_size
                    SimulatedObservation!(models[i, j](src[i]), rcv_sim[i, j])
                    push!(x[k], reshape(rcv_sim[i, j].rcvv, 1, -1))
                end
                x[k] = vcat(x[k]...)
            end
            for (k,i) in enumerate(src_ids)
                for j = 1:sample_size
                    push!(y[k], reshape(Rs[i, j], 1, -1))
                end
                y[k] = vcat(y[k]...)
            end

            ds = Array{PyObject}(undef, length(src_ids))
            for i = 1:length(src_ids)
                ds[i] = dist(x[i], y[i], 1)
                # @info ds[i]
            end
        end
        return ds
    end

    n_gpu = min(length(use_gpu()), nsrc)

    if n_gpu==0
        @error "You must use GPU for sampling_compute_loss_and_grads_GPU_v2"
    else
        ds = PyObject[]
        for i = 1:n_gpu
            jobs = [k for k=1:nsrc if k%n_gpu==i-1]
            println("GPU $i --> sources $jobs with $(batch_size) vp each")
            [push!(ds, d) for d in run_on_gpu_device!(i-1, jobs)]
        end
        @assert length(ds)==nsrc
    end

    @pywith tf.device("/cpu:0") begin
        a = tf.ones((size(ds[1],1),), dtype=tf.float64)/Float64(size(ds[1],1))
        b = tf.ones((size(ds[1],2),), dtype=tf.float64)/Float64(size(ds[1],2))
        loss = constant(0.0)
        for i = 1:nsrc
            loss += sinkhorn(a, b, ds[i], reg=reg, method=method)
        end
    end
    return loss

end



"""

Mathcing statistics. 
- `models` : nsrc * batch_size
- `src` : nsrc
- `rcv_sim` : nsrc * batch_size
- `Rs` : nsrc * sample_size
"""
function sampling_compute_loss_and_grads_GPU_v3(models, src, rcv_sim, Rs; reg::Float64=1.0, method::String="sinkhorn")
    
    local loss
    nsrc, batch_size = size(models)
    @assert nsrc == length(src)
    @assert size(rcv_sim,1) == nsrc
    @assert size(rcv_sim,2) == batch_size
    @assert size(Rs,1)==nsrc
    sample_size = size(Rs,2)

    function run_on_gpu_device!(gpu_id, src_ids)
        local ds
        @pywith tf.device("/gpu:$(gpu_id)") begin
            x = Any[[] for i=1:length(src_ids)]
            y = Any[[] for i=1:length(src_ids)]
            for (k,i) in enumerate(src_ids)
                for j = 1:batch_size
                    SimulatedObservation!(models[i, j](src[i]), rcv_sim[i, j])
                    push!(x[k], reshape(rcv_sim[i, j].rcvv, 1, -1))
                end
                x[k] = vcat(x[k]...)
            end
            for (k,i) in enumerate(src_ids)
                for j = 1:sample_size
                    push!(y[k], reshape(Rs[i, j], 1, -1))
                end
                y[k] = vcat(y[k]...)
            end

            ds = Array{PyObject}(undef, length(src_ids))
            for i = 1:length(src_ids)
                ds[i] = sum((mean(x[i], dims=1) - mean(y[i], dims=1))^2 +
                                 (std(x[i], dims=1) - std(y[i], dims=1))^2)
                @info ds[i]
            end
        end
        return ds
    end

    n_gpu = min(length(use_gpu()), nsrc)

    if n_gpu==0
        @error "You must use GPU for sampling_compute_loss_and_grads_GPU_v2"
    else
        ds = PyObject[]
        for i = 1:n_gpu
            jobs = [k for k=1:nsrc if k%n_gpu==i-1]
            println("GPU $i --> sources $jobs with $(batch_size) vp each")
            [push!(ds, d) for d in run_on_gpu_device!(i-1, jobs)]
        end
        @assert length(ds)==nsrc
    end

    @pywith tf.device("/cpu:0") begin
        # a = tf.ones((size(ds[1],1),), dtype=tf.float64)/Float64(size(ds[1],1))
        # b = tf.ones((size(ds[1],2),), dtype=tf.float64)/Float64(size(ds[1],2))
        loss = constant(0.0)
        for i = 1:nsrc
            loss += ds[i]
        end
    end
    return loss
end

function compute_forward_GPU(model::Function, src::Union{Array{AcousticSource},Array{ElasticSource}},
    rcv::Union{Array{AcousticReceiver}, Array{ElasticReceiver}})

    function run_on_gpu_device(gpu_id, jobs)
        local rcvv
        @pywith tf.device("/gpu:$(gpu_id)") begin
            [SimulatedObservation!(model(src[i]), rcv[i]) for i = jobs]
            rcvv = [rcv[i].rcvv for i = jobs]
        end
        return rcvv
    end

    n = length(src)
    n_gpu = min(length(use_gpu()), n)
    if n_gpu==0
        error("No GPU resources available")
    end

    Rs = Array{PyObject}(undef, n)
    for i = 1:n_gpu
        jobs = [k for k=1:n if k%n_gpu==i-1]
        println("GPU $i --> sources $jobs")
        Rs[jobs] = run_on_gpu_device(i-1, jobs)
    end

    return Rs
end


function variable_source(param::Union{ElasticPropagatorParams,AcousticPropagatorParams}, 
        x::Union{PyObject, Float64}, y::Union{PyObject, Float64}, v::Union{Array{Float64},PyObject}; 
        sigma::Union{Float64, Nothing} = nothing)
    x = convert_to_tensor(x) 
    y = convert_to_tensor(y) 
    v = convert_to_tensor(v)
    if isnothing(sigma)
        sigma = 1.0
    end
    srci = zeros(Int64, (param.NX+2)*(param.NY+2))
    srcj = zeros(Int64, (param.NX+2)*(param.NY+2))
    xs = zeros((param.NX+2)*(param.NY+2))
    ys = zeros((param.NX+2)*(param.NY+2))
    k = 1
    for i = 1:param.NX+2
        for j = 1:param.NY+2
            srci[k] = i 
            srcj[k] = j
            xs[k] = (i-1)*param.DELTAX
            ys[k] = (j-1)*param.DELTAY
            k += 1
        end
    end
    # magn = 1/(2π*sigma) * exp( - ((xs-x)^2 + (ys-y)^2)/2sigma^2)
    h = param.DELTAX
    # srcv = reshape(v, length(v), 1) * exp(-((xs-x)^2+(ys-y)^2)/(h^2))'
    magn = 1/(2π*sigma) * exp( - (((xs-x)/(sqrt(2sigma)*param.DELTAX))^2 +
                    ((ys-y)/(sqrt(2sigma)*param.DELTAY))^2))
    srcv = reshape(v, length(v), 1) * magn'
    return srci, srcj, srcv
end


function variable_source(param::AcousticPropagatorParams, 
    ix::Array{Int64,1}, iy::Array{Int64,1}; name::String = "src", source_scale::Float64 = 1.0)
    @assert length(ix)==length(iy)
    v = Variable(zeros(param.NSTEP+1, length(ix)), name=name) * source_scale
    AcousticSource(ix, iy, v)
end