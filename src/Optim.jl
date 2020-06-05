export LBFGS!


@doc raw"""
    LBFGS!(sess::PyObject, loss::PyObject, max_iter::Int64=15000; 
    vars::Array{PyObject}=PyObject[], callback::Union{Function, Nothing}=nothing, kwargs...)
`BFGS!` is a simplified interface for BFGS optimizer. See also [`ScipyOptimizerInterface`](@ref).
`callback` is a callback function with signature 
```julia
callback(vs::Array, iter::Int64, loss::Float64)
```
`vars` is an array consisting of tensors and its values will be the input to `vs`.
# Example 1
```julia
a = Variable(1.0)
loss = (a - 10.0)^2
sess = Session(); init(sess)
BFGS!(sess, loss)
```
# Example 2
```julia
θ1 = Variable(1.0)
θ2 = Variable(1.0)
loss = (θ1-1)^2 + (θ2-2)^2
cb = (vs, iter, loss)->begin 
    printstyled("[#iter $iter] θ1=$(vs[1]), θ2=$(vs[2]), loss=$loss\n", color=:green)
end
sess = Session(); init(sess)
cb(run(sess, [θ1, θ2]), 0, run(sess, loss))
BFGS!(sess, loss, 100; vars=[θ1, θ2], callback=cb)
```
# Example 3
Use `bounds` to specify upper and lower bound of a variable. 
```julia
x = Variable(2.0)    
loss = x^2
sess = Session(); init(sess)
BFGS!(sess, loss, bounds=Dict(x=>[1.0,3.0]))
```
"""->
function LBFGS!(sess::PyObject, loss::PyObject, max_iter::Int64=15000; 
    vars::Array{PyObject}=PyObject[], callback::Union{Function, Nothing}=nothing, kwargs...)
    __cnt = 0
    __loss = 0
    __var = nothing
    out = []
    function print_loss(l, vs...)
        if !isnothing(callback); __var = vs; end
        if mod(__cnt,1)==0
            println("iter $__cnt, current loss=",l)
        end
        __loss = l
        __cnt += 1
    end
    __iter = 0
    function step_callback(rk)
        if mod(__iter,1)==0
            println("================ STEP $__iter ===============")
        end
        if !isnothing(callback)
            callback(__var, __iter, __loss)
        end
        push!(out, __loss)
        __iter += 1
    end
    kwargs = Dict(kwargs)
    if haskey(kwargs, :bounds)
        kwargs[:var_to_bounds] = kwargs[:bounds]
        delete!(kwargs, :bounds)
    end
    if haskey(kwargs, :var_to_bounds)
        desc = "`bounds` or `var_to_bounds` keywords of `BFGS!` only accepts dictionaries whose keys are Variables"
        for (k,v) in kwargs[:var_to_bounds]
            if !(hasproperty(k, "trainable"))
                error("The tensor $k does not have property `trainable`, indicating it is not a `Variable`. $desc\n")
            end 
            if !k.trainable
                @warn("The variable $k is not trainable, ignoring the bounds")
            end
        end
    end
    opt = ScipyOptimizerInterface(loss, method="L-BFGS-B",options=Dict("maxiter"=> max_iter, "ftol"=>1e-12, "gtol"=>1e-12); kwargs...)
    @info "Optimization starts..."
    ScipyOptimizerMinimize(sess, opt, loss_callback=print_loss, step_callback=step_callback, fetches=[loss, vars...])
    out
end

"""
    LBFGS!(value_and_gradients_function::Function, initial_position::Union{PyObject, Array{Float64}}, max_iter::Int64=50, args...;kwargs...)
Applies the BFGS optimizer to `value_and_gradients_function`
"""
function LBFGS!(value_and_gradients_function::Function, 
    initial_position::Union{PyObject, Array{Float64}}, max_iter::Int64=50, args...;kwargs...)
    tfp.optimizer.bfgs_minimize(value_and_gradients_function, 
        initial_position=initial_position, args...;max_iterations=max_iter, kwargs...)[5]
end


@doc raw"""
    LBFGS!(sess::PyObject, loss::PyObject, grads::Union{Array{T},Nothing,PyObject}, 
    vars::Union{Array{PyObject},PyObject}; kwargs...) where T<:Union{Nothing, PyObject}
Running BFGS algorithm
``\min_{\texttt{vars}} \texttt{loss}(\texttt{vars})``
The gradients `grads` must be provided. Typically, `grads[i] = gradients(loss, vars[i])`. 
`grads[i]` can exist on different devices (GPU or CPU). 
# Example 1
```julia
import Optim # required
a = Variable(0.0)
loss = (a-1)^2
g = gradients(loss, a)
sess = Session(); init(sess)
BFGS!(sess, loss, g, a)
```
# Example 2
```julia 
import Optim # required
a = Variable(0.0)
loss = (a^2+a-1)^2
g = gradients(loss, a)
sess = Session(); init(sess)
cb = (vs, iter, loss)->begin 
    printstyled("[#iter $iter] a = $vs, loss=$loss\n", color=:green)
end
BFGS!(sess, loss, g, a; callback = cb)
```
"""
function LBFGS!(sess::PyObject, loss::PyObject, grads::Union{Array{T},Nothing,PyObject}, 
        vars::Union{Array{PyObject},PyObject}; callback::Union{Function, Missing}=missing, kwargs...) where T<:Union{Nothing, PyObject}

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
    __value = nothing
    function f(x)
        run(sess, assign_ops, pl=>x)
        __loss = run(sess, loss)
        return __loss
    end

    function g!(G, x)
        run(sess, assign_ops, pl=>x)
        __value = x
        G[:] = run(sess, grds)
    end

    function callback1(x)
        __iter = x.iteration
        __loss = x.value
        push!(__losses, __loss)
        if !ismissing(callback)
            callback(__value, __iter, __loss)
        end
        false
    end

    Optim.optimize(f, g!, x0, Optim.LBFGS(), Optim.Options(show_trace=true, callback=callback1, kwargs...))
    return __losses
end


