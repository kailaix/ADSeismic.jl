export LBFGS!

@doc raw"""
    LBFGS!(sess::PyObject, loss::PyObject, grads::Union{Array{T},Nothing,PyObject}, 
        vars::Union{Array{PyObject},PyObject}; IT_PLOT::Int64 = 10, NX::Int64, NY::Int64, kwargs...) where T<:Union{Nothing, PyObject}

L-BFGS algorithm for optimizing variables. 
- `sess`: A working session for current computation
- `loss`: Loss function
- `grads`: gradients ``\frac{\partial L}{\partial v}``, where ``v`` is the unknown
- `vars`: unknown variables
- `max_iter`: maximum number of iterations

## Keyword arguments
- `IT_PLOT`: the plotting frequency. 
- `DIR`: the directory for saving the plots. By default, `DIR=tmp`
- `NX`, `NY`: the numbers of DOFs in ``x`` and ``y`` direction
- Additional keyword arguments are passed to `Optim.optimize`
"""
function LBFGS!(sess::PyObject, loss::PyObject, grads::Union{Array{T},Nothing,PyObject}, 
    vars::Union{Array{PyObject},PyObject}; 
    callback::Union{Function, Nothing}=nothing, max_iter::Int64=15000,
    kwargs...) where T<:Union{Nothing, PyObject}
    if isa(grads, PyObject); grads = [grads]; end
    if isa(vars, PyObject); vars = [vars]; end
    if length(grads)!=length(vars); error("ADCME: length of grads and vars do not match"); end

    idx = ones(Bool, length(grads))
    for i = 1:length(grads)
        if isnothing(grads[i])
            idx[i] = false
        end
    end
    grads = grads[idx]
    vars = vars[idx]

    sizes = []
    for v in vars
        push!(sizes, size(v))
    end
    grds = vcat([tf.reshape(g, (-1,)) for g in grads]...)
    vs = vcat([tf.reshape(v, (-1,)) for v in vars]...); x0 = run(sess, vs)
    pl = placeholder(x0)
    n = 0
    assign_ops = PyObject[]
    for (k,v) in enumerate(vars)
        push!(assign_ops, assign(v, tf.reshape(pl[n+1:n+prod(sizes[k])], sizes[k])))
        n += prod(sizes[k])
    end

    __loss = 0.0
    __losses = Float64[]
    __iter = 0
    function f(x)
        run(sess, assign_ops, pl=>x)
        __loss = run(sess, loss)
        __iter += 1
        if !isnothing(callback)
            callback(x, __iter, __loss)
        end
        # __iter += 1
        return __loss
    end

    function g!(G, x)
        run(sess, assign_ops, pl=>x)
        G[:] = run(sess, grds)
    end

    function callback_optim(x)
        push!(__losses, __loss)
        if __iter>max_iter
            return true
        end
        false
    end

    Optim.optimize(f, g!, x0, Optim.LBFGS(), Optim.Options(show_trace=true, callback=callback_optim,
        kwargs...))
    return __losses
end



function LBFGS!(sess::PyObject, loss::PyObject, max_iter::Int64=15000; vars::Array{PyObject}=PyObject[],
                     callback::Union{Nothing,Function}=nothing, kwargs...)
    __cnt = 0
    __loss = 0
    out = []
    __v = nothing
    function print_loss(l, vars...)
        __v = vars
        if mod(__cnt,1)==0
            @printf("iter %d, loss=%e\n", __cnt, l)
        end
        __loss = l
        __cnt += 1
    end
    __iter = 0
    function step_callback(rk)
        if mod(__iter,1)==0
            println("================ ITER $__iter ===============")
        end
        if !isnothing(callback)
            callback(__v, __iter, __loss)
        end
        push!(out, __loss)
        __iter += 1
    end
    opt = ScipyOptimizerInterface(loss, method="L-BFGS-B",
        options=Dict("maxiter"=> max_iter, "ftol"=>1e-12, "gtol"=>1e-30))
    @info "Optimization starts..."
    ScipyOptimizerMinimize(sess, opt, loss_callback=print_loss, step_callback=step_callback, fetches=[loss, vars...])
    out
end