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

R_ = constant(R)

idx_var = pml+Int(round(0.1/Δx))

inv_sigmoid(y) = @. - log(1.0/y - 1.0)

case = "fwi"

## fwi
if case == "fwi"
if model == "step"
    var= Variable(zeros(N-1-idx_var)) + inv_sigmoid(1.3/2.0)
    var = -2.0*sigmoid(var)
elseif model == "slop"
    var = Variable(zeros(N-1-idx_var))
end
end

## FC
if case == "nnfwi"
x0 = collect( range(0, 1, length=N-1-idx_var) )
init_guess = fc_init([1, 30, 30, 30, 1])
θ = Variable(init_guess)
if model == "step"
    var= fc(x0, [30, 30, 30, 1], θ) * 0.1 + inv_sigmoid(1.3/2.0)
    var = -2.0*sigmoid(var)
elseif model == "slop"
    var= fc(x0, [30, 30, 30, 1], θ) * 0.1 
end
end

## CNN
# x = constant(randn(1,4,1))
# x = tf.keras.layers.Conv1D(8, 3, padding="same")(x)
# x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
# x = tf.keras.layers.UpSampling1D(4)(x)
# x = tf.keras.layers.Conv1D(8, 3, padding="same")(x)
# x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
# x = tf.keras.layers.UpSampling1D(4)(x)
# x = tf.keras.layers.Conv1D(8, 3, padding="same")(x)
# x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
# x = tf.keras.layers.UpSampling1D(4)(x)
# x = tf.keras.layers.Conv1D(1, 3, padding="same")(x)
# x = squeeze(x)
# var = tf.slice(x, (length(x) - (N-1-idx_var),).÷2, (N-1-idx_var,)) * 0.1 - 1.0
# var = (3.5 + 0.5) * sigmoid(var) - 0.5


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
loss1 = sum((E[:, idx_rcv[1]] - E_true[:, idx_rcv[1]])^2 * 1.0)
loss2 = sum((E[:, idx_rcv[2]] - E_true[:, idx_rcv[2]])^2 * 1e-3)

if case == "fwi"
    # loss += 1e3*reg_smooth + 0.0*reg_TV
    # case = string(case, "_reg_smooth")
    loss += 1e2*reg_smooth + 1e2*reg_TV
    case = string(case, "_reg1e2_smooth_TV")
end


if !ispath(joinpath("figures", model, case))
    mkpath(joinpath("figures", model, case))
end
if !ispath(joinpath("results", model, case))
    mkpath(joinpath("results", model, case))
end

function cb(vs, iter, loss)
    if (mod(iter, 20) == 1)
        E_est = vs[1]
        cH_est = vs[2]
        @info iter, vs[3], vs[4]
        close("all")
        figure(figsize = (10,6))
        subplot(221)
        plot(xH, cH_true, "C0", label="True")
        plot(xH, cH0, "k--", label="Initial")
        plot(xH, cH_est, "C3--", label="Estimated")
        legend()
        # xlim([0,1])
        autoscale(enable=true, axis="x", tight=true)
        xlabel("Location (km)")
        ylabel("Velocity (km/s)")
        subplot(222)
        plot((0:NT)*Δt, E_true[:,idx_rcv[1]], "C0", label="True #1",  alpha=0.8)
        plot((0:NT)*Δt, E_true[:,idx_rcv[2]], "C2", label="True #2", alpha=0.8)
        plot((0:NT)*Δt, E_est[:,idx_rcv[1]], "C3--",  label="Estimated #1", alpha=0.8)
        plot((0:NT)*Δt, E_est[:,idx_rcv[2]], "C1--",  label="Estimated #2", alpha=0.8)
        legend(loc="upper right")
        gca().ticklabel_format(axis="y", style="sci", scilimits=(0,0))
        autoscale(enable=true, axis="x", tight=true)
        xlabel("Time (s)")
        ylabel("Amplitude")
        subplot(223)
        pcolormesh(xE, (0:NT)*Δt, E_true, cmap="seismic", vmax=maximum(E_true)/3.0, vmin=-maximum(E_true)/3.0, rasterized=true)
        plot(xE[idx_src]*ones(NT+1), (0:NT)*Δt, "k", label="Source", alpha=0.8)
        plot(xE[idx_rcv[1]]*ones(NT+1), (0:NT)*Δt, "C0--", label="Receiver #1", alpha=0.8)
        plot(xE[idx_rcv[2]]*ones(NT+1), (0:NT)*Δt, "C2--", label="Receiver #2", alpha=0.8)
        # xlim([0,1])
        autoscale(enable=true, axis="x", tight=true)
        legend(loc="upper center")
        xlabel("Localtion (km)")
        ylabel("Time (s)")
        title("True")
        subplot(224)
        pcolormesh(xE, (0:NT)*Δt, E_est, cmap="seismic", vmax=maximum(E_true)/3.0, vmin=-maximum(E_true)/3.0, rasterized=true)
        plot(xE[idx_src]*ones(NT+1), (0:NT)*Δt, "k", label="Source", alpha=0.8)
        plot(xE[idx_rcv[1]]*ones(NT+1), (0:NT)*Δt, "C0--", label="Receiver #1", alpha=0.8)
        plot(xE[idx_rcv[2]]*ones(NT+1), (0:NT)*Δt, "C2--", label="Receiver #2", alpha=0.8)
        # xlim([0,1])
        autoscale(enable=true, axis="x", tight=true)
        legend(loc="upper center")
        xlabel("Localtion (km)")
        ylabel("Time (s)")
        title("Estimated")

        # subplot(325)
        # plot((0:NT)*Δt, E_true[:,idx_rcv[1]].-E_est[:,idx_rcv[1]], "C0", label="Diff #1",  alpha=0.8)
        # legend(loc="upper right")
        # gca().ticklabel_format(axis="y", style="sci", scilimits=(0,0))
        # autoscale(enable=true, axis="x", tight=true)
        # xlabel("Time (s)")
        # ylabel("Amplitude")
        # subplot(326)
        # plot((0:NT)*Δt, E_true[:,idx_rcv[2]].-E_est[:,idx_rcv[2]], "C3", label="Diff #2", alpha=0.8)
        # legend(loc="upper right")
        # gca().ticklabel_format(axis="y", style="sci", scilimits=(0,0))
        # autoscale(enable=true, axis="x", tight=true)
        # xlabel("Time (s)")
        # ylabel("Amplitude")

        tight_layout()
        savefig(joinpath("figures", model, case, "fwi$(lpad(iter,5,"0")).png"), bbox_inch="tight")
        if (mod(iter, 200) == 1) 
        matwrite(joinpath("results", model, case, "fwi$(lpad(iter,5,"0")).mat"), 
                 Dict("idx_rcv"=>idx_rcv, "idx_src"=>idx_src, "xE"=>xE, 
                      "cH0"=>cH0, "cH_true"=>cH_true, "cH_est" => cH_est, 
                      "E_true" => E_true, "E_est" => E_est))
        # writedlm(joinpath("results", model, case, "fwi$(lpad(iter,5,"0")).txt"), E_est)
        end
    end
end


lr = placeholder(Float64, shape=[])
opt = AdamOptimizer(learning_rate=lr).minimize(loss)
# opt = tf.train.AdagradOptimizer(learning_rate=0.01).minimize(loss)
# opt = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)
# opt = tf.train.RMSPropOptimizer(learning_rate=0.01).minimize(loss)

sess = Session(); init(sess)
cH0 = run(sess, cH)
cb(run(sess, [E, cH, loss1, loss2]), -1, run(sess, loss))
BFGS!(sess, loss, vars = [E, cH, loss1, loss2], callback = cb)


# for iter = 0:10000
#     _, l = run(sess, [opt, loss], feed_dict=Dict(lr=>(randexp()*0.001+0.00001)))
#     cb(run(sess, [E, cH]), iter, l)
#     if mod(iter, 20) == 1
#         @info iter, l
#     end
# end

