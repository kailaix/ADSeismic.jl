export  load_params, load_elastic_model, load_elastic_receiver, load_elastic_source,
        load_acoustic_model, load_acoustic_receiver, load_acoustic_source

function load_params(filename::String, option::String="Acoustic"; kwargs...)

    d =  matread(filename)

    if option == "Acoustic"
        param = AcousticPropagatorParams(NX=d["nx"]-2, NY=d["ny"]-2,
                        NSTEP=d["nt"], DELTAX=d["dx"], 
                        DELTAY=d["dy"], DELTAT=d["dt"], kwargs...)
    elseif option == "Elastic"
        param = ElasticPropagatorParams(NX=d["nx"]-2, NY=d["ny"]-2,
                        NSTEP=d["nt"], DELTAX=d["dx"], 
                        DELTAY=d["dy"], DELTAT=d["dt"], kwargs...)
    else
        error("option should be either Acoustic or Elastic!");
    end

    return param
end

function load_elastic_model(filename::String; 
        inv_vp::Bool = false, inv_vs::Bool = false, inv_rho::Bool = false,
        kwargs...)
    d = matread(filename)
    param = ElasticPropagatorParams(NX=d["nx"]-2, NY=d["ny"]-2,
                        NSTEP=d["nt"], DELTAX=d["dx"], 
                        DELTAY=d["dy"], DELTAT=d["dt"], 
                        vp_ref=mean(d["vp"]), f0=d["f0"]/2; kwargs...)

    vp = constant_or_variable(d["vp"], trainable=inv_vp, name="vp")
    vs = constant_or_variable(d["vs"], trainable=inv_vs, name="vs")
    ρ = constant_or_variable(d["rho"], trainable=inv_rho, name="rho")
    λ, μ, ρ = compute_properties(vp, vs, ρ)
    ep_fun = x->ElasticPropagatorSolver(param, x, ρ, λ, μ)
    @info "Successfully load elastic model!"
    ep_fun
end


function load_elastic_receiver(filename::String)
    d = matread(filename)
    rcv = d["receiver"]
    if length(rcv)==0
        return ElasticReceiver[]
    end
    out = Array{ElasticReceiver}(undef, length(rcv))
    for i = 1:length(rcv)
        out[i] = ElasticReceiver(safe_vec(rcv[i]["ix"]), safe_vec(rcv[i]["iy"]), safe_vec(rcv[i]["type"]))
    end
    @info "Successfully load elastic receiver!"
    out
end

function load_elastic_source(filename::String)
    d = matread(filename)
    src = d["source"]
    out = Array{ElasticSource}(undef, length(src))
    for i = 1:length(src)
        if length(size(src[i]["vec"]))==1
            src[i]["vec"] = reshape(src[i]["vec"], :, 1)
        end
        if d["nt"] != size(src[i]["vec"], 1) && d["nt"] == size(src[i]["vec"], 2)
            src[i]["vec"] = Array(src[i]["vec"]')
        end
        @assert d["nt"] == size(src[i]["vec"], 1)
        out[i] = ElasticSource(safe_vec(src[i]["ix"]), safe_vec(src[i]["iy"]), safe_vec(src[i]["type"]), src[i]["vec"])
    end
    @info "Successfully load elastic source!"
    out
end


"""
load_acoustic_model(filename::String; inv_vp::Bool = false, kwargs...)

Load acoustic model from a MAT file.

# Arguments
- 'filename::String': MAT filename
- 'inv_vp::Bool': inversion of velocity (default: false)

# Return
- AcousticPropagatorSolver(src::AcousticSource)
"""
function load_acoustic_model(filename::String; inv_vp::Bool = false, kwargs...)
    d = matread(filename)
    kwargs = Dict{Any, Any}(kwargs)
    mask = missing
    if haskey(kwargs, :mask)
        mask = kwargs[:mask]
        delete!(kwargs, :mask)
    end
    if haskey(kwargs, :vp)
        vp = kwargs[:vp]
        delete!(kwargs, :vp)
    end
    param = AcousticPropagatorParams(NX=d["nx"]-2, NY=d["ny"]-2,
                        NSTEP=d["nt"], DELTAX=d["dx"], 
                        DELTAY=d["dy"], DELTAT=d["dt"], kwargs...)
    if (@isdefined vp)
        ap_fun = x->AcousticPropagatorSolver(param, x, vp^2)
    else
        vp = constant_or_variable(d["vp"], trainable = inv_vp, name="vp", mask=mask)^2
        ap_fun = x->AcousticPropagatorSolver(param, x, vp)
    end
    @info "Successfully load acoustic model!"
    ap_fun
end


function load_acoustic_receiver(filename::String)
    d = matread(filename)
    rcv = d["receiver"]
    if length(rcv)==0
        return AcousticReceiver[]
    end
    out = Array{AcousticReceiver}(undef, length(rcv))
    for i = 1:length(rcv)
        out[i] = AcousticReceiver(safe_vec(rcv[i]["ix"]), safe_vec(rcv[i]["iy"]), missing)
    end
    @info "Successfully load acoustic receiver!"
    out
end

function load_acoustic_source(filename::String)
    d = matread(filename)
    src = d["source"]
    out = Array{AcousticSource}(undef, length(src))
    for i = 1:length(src)
        if length(size(src[i]["vec"]))==1 || size(src[i]["vec"], 1) == 1
            src[i]["vec"] = reshape(src[i]["vec"], :, 1)
        end
        @assert d["nt"] == size(src[i]["vec"], 1)
        out[i] = AcousticSource(safe_vec(src[i]["ix"]), safe_vec(src[i]["iy"]), src[i]["vec"])
    end
    @info "Successfully load acoustic source!"
    out
end

function safe_vec(a)
    if isa(a, Number)
        return [a]
    else
        return a[:]
    end
end


function constant_or_variable(x::Array{Float64,2}; trainable::Bool=false, name::String, mask::Union{Array{Float64}, Missing}=missing)    
    # @show mean(x)
    # x = ones(size(x)...)*mean(x)
    # return constant(x)
    if trainable
        meanx = mean(x)
        x = x/meanx
        x_ = Variable(x, name=name) 
        if !ismissing(mask)
            x_ =  mask .* x_ +  x .* (1 .- mask)
        end
        return x_ * meanx
        # mask = zero(x)
        # mask[:, 15:end] .= 1.0 #marmousi
        # mask[:, 17:end-16] .= 1.0 #layer model
        # mask[:, 15:end-14] .= 1.0 #layer model
        # mask[15:end-14, 15:end-14] .= 1.0 #layer model
        # mask[12:end-11, 17:end-16] .= 1.0 #layer model
        # mask[:, :] .= 1.0
        # x_mask =  mask .* x_ +  x .* (1 .- mask)
        # @show x_mask[1,1]
        # return  x_mask * meanx
    else
        return constant(x)
    end
end

function Base.:get(s::String)
    return tensor(s*":0")
end