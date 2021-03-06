using ADCME
using PyPlot 
using Random
using MAT
using DelimitedFiles
matplotlib.use("agg")

include("fwi_data.jl")
reset_default_graph()
tf.set_random_seed(123)
tf.random.set_random_seed(123)
θ0 = matread("results/weight$model.mat")["theta"][:]

μ = Variable(θ0)
ρ = Variable(zeros(length(θ0)))


if model=="step"
    σ = log(1 + exp(ρ))
else
    σ = 0.1*log(1 + exp(ρ))
end

# dt = 1e-10
dt = 1e-8
# dt = 0

function likelihood(z)
    θ = μ + z * σ


    R_ = constant(R)

    idx_var = pml+Int(round(0.1/Δx))

    inv_sigmoid(y) = @. - log(1.0/y - 1.0)

    # case = "fwi"
    case = "nnfwi"

    ## FC
    if case == "nnfwi"
    x0 = collect( range(0, 1, length=N-1-idx_var) )
    if model == "step"
        var= fc(x0, [30, 30, 30, 1], θ) * 0.1 + inv_sigmoid(1.3/2.0)
        var = -2.0*sigmoid(var)
    elseif model == "slop"
        var= fc(x0, [30, 30, 30, 1], θ) * 0.1 
    end
    end

    cH = scatter_add(constant(cH_init), idx_var+1:N-1, var)

    cE = (cH[1:end-1]+cH[2:end])/2

    reg_TV = sum(abs(cH[2:end]-cH[1:end-1]))
    reg_smooth = sum((cH[1:end-2] + cH[3:end] - 2*cH[2:end-1])^2)

    function condition(i, E_arr, H_arr)
        i<=NT+1
    end

    function body(i, E_arr, H_arr)
        E = read(E_arr, i-1)
        H = read(H_arr, i-1)
        
        ΔH = cH * (E[2:end]-E[1:end-1])/Δx - σH*H
        H += ΔH * Δt
        ΔH = 1/(24Δx) * cH[2:N-2] * (-E[4:end] + 3E[3:end-1] - 3E[2:end-2] + E[1:end-3]) 
        H = scatter_add(H, 2:N-2, ΔH * Δt)
        
        ΔE = cE * (H[2:end]-H[1:end-1])/Δx - σE[2:end-1]*E[2:end-1] + R_[i] * Z
        E = scatter_add(E, 2:N-1, ΔE * Δt)
        ΔE = 1/(24Δx) * cE[2:end-1] * (-H[4:end] + 3H[3:end-1] - 3H[2:end-2] + H[1:end-3]) 
        E = scatter_add(E, 3:N-2, ΔE*Δt)

        i+1, write(E_arr, i, E), write(H_arr, i, H)
    end

    E_arr = TensorArray(NT+1)
    H_arr = TensorArray(NT+1)

    E_arr = write(E_arr, 1, zeros(N))
    H_arr = write(H_arr, 1, zeros(N-1))

    i = constant(2, dtype = Int32)

    _, E, H = while_loop(condition, body, [i, E_arr, H_arr])

    E = stack(E); E = set_shape(E, (NT+1, N))
    H = stack(H)

    loss = sum((E[:, idx_rcv] - E_true[:, idx_rcv])^2 * [1.0, 1e-3])

    lkh = dt*loss - sum((θ-μ)^2/σ^2) + sum((θ-θ0)^2)
    cH, lkh
end

z = tf.random_normal((8, length(θ0)), dtype = tf.float64)
# z = constant(randn(8, length(θ0)))
loss = constant(0.0)
for i = 1:size(z, 1)
    global cH, loss_ = likelihood(z[i])
    global loss += loss_ 
end
lr = placeholder(Float64, shape = [])
opt = AdamOptimizer(lr).minimize(loss)

sess = Session(); init(sess)

if dt>0.0
    global losses = []
    for i = 1:1000
        _, l = run(sess, [opt, loss], lr=>0.001)
        push!(losses, l)
        @info i, l 
    end
    close("all")
    plot(losses)
    savefig("loss.png")
end 
# error()
close("all")
cH, _ = likelihood(tf.random_normal((length(θ0),), dtype = tf.float64))
cH_est = zeros(length(cH), 100)
for i = 1:100
    cH_est[:,i] = run(sess, cH)
    plot(xH, cH_est[:,i], "--", color="gray", alpha = 0.5)
end
plot(xH, cH_true, "C0", label="True")
# plot(xH, cH0, "k--", label="Initial")
legend()

if dt≈0
    savefig("$(model)_prior.png")
else 
    savefig("$model.png")
end
