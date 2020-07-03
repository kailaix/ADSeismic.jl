using ADCME
using PyPlot 
using ADCMEKit 

n = 100
pml = 30
C = 1000.0
# C = 0.0
NT = 2000
Δt = 2.0/NT 
x0 = LinRange(0, 1, n+1)
h = 1/n 
xE = Array((0:n+2pml)*h .- pml*h)
xH = (xE[2:end]+xE[1:end-1])/2 
N = n + 2pml + 1


σE = zeros(N)
for i = 1:pml
    d = i*h
    σE[pml + n + 1 + i] = C* (d/(pml*h))^3
    σE[pml+1-i] = C* (d/(pml*h))^3
end

σH = zeros(N-1)
for i = 1:pml 
    d = (i-1/2)*h 
    σH[pml + n + i] = C* (d/(pml*h))^3
    σH[pml+1-i] = C* (d/(pml*h))^3
end

function ricker(dt = 0.002, f0 = 5.0)
    nw = 2/(f0*dt)
    nc = floor(Int, nw/2)
    t = dt*collect(-nc:1:nc)
    b = (π*f0*t).^2
    # w = @. (1 - 2b)*exp(-b)
    w = @. exp(-b)
end
R = ricker()
if length(R)<NT+1
    R = [R;zeros(NT+1-length(R))]
end
R = R[1:NT+1]
# tn = ( 0:NT ) *Δt
# R = @. exp( -20(tn-0.3)^2)
# error()
R_ = constant(R)


cH = ones(length(xH)) * 2.
cH[pml + Int(round(0.5÷h)): pml + Int(round(0.75÷h))] .= 1.0
cH[pml + Int(round(0.75÷h)): end] .= 1.

mask = ones(length(xH))
mask[pml+1:end-pml] .= 0.0
c_ = Variable( ones(N-1) * 2)

x0 = collect(((pml:N-pml -1) .- pml)*h)
# c_ = squeeze(fc(x0, [20,20,1])) + 1.0
# c_ = 2tanh(c_) + 2.5
# cH = scatter_update(constant(cH), pml+1:N-pml, c_)

cH = c_
# cH = constant(cH)
cE = (cH[1:end-1]+cH[2:end])/2
Z = zeros(N)
Z[pml + pml÷2] = 1.
Z = Z[2:end-1]

function condition(i, E_arr, H_arr)
    i<=NT+1
end

function body(i, E_arr, H_arr)
    E = read(E_arr, i-1)
    H = read(H_arr, i-1)
    
    ΔH = cH * (E[2:end]-E[1:end-1])/h - σH*H
    H += ΔH * Δt
    ΔH = 1/(24h) * cH[2:N-2] * (-E[4:end] + 3E[3:end-1] - 3E[2:end-2] + E[1:end-3]) 
    H = scatter_add(H, 2:N-2, ΔH * Δt)
    
    
    ΔE = cE * (H[2:end]-H[1:end-1])/h - σE[2:end-1]*E[2:end-1] + R_[i] * Z
    E = scatter_add(E, 2:N-1, ΔE * Δt)
    ΔE = 1/(24h) * cE[2:end-1] * (-H[4:end] + 3H[3:end-1] - 3H[2:end-2] + H[1:end-3]) 
    E = scatter_add(E, 3:N-2, ΔE*Δt)


    # E = scatter_update(E, N÷2, R_[i])
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

loss = sum((E[:, [pml+pml÷2;N-2pml]] - E_[:, [pml+pml÷2;N-2pml]])^2)

# loss = sum((E-E_)^2)
# loss = sum(E^2)

function cb(vs, iter, loss)
    if mod(iter, 20)!=1
        return false
    end
    est_E_ = vs[1]
    est_cH = vs[2]
    close("all")
    figure(figsize = (14,4))
    subplot(131)
    pcolormesh(xE, (0:NT)*Δt, est_E_-E_)
    plot(xE[pml + pml÷2] * ones(NT+1) , (0:NT)*Δt, "r", label="Source/Receiver")
    xlabel("Localtion")
    ylabel("Time")
    subplot(132)
    plot((0:NT)*Δt, est_E_[:, pml + pml÷2] - E_[:, pml+pml÷2])
    xlabel("Time")
    ylabel("Signal")
    subplot(133)
    plot(xH, est_cH)
    xlabel("Location")
    ylabel("\$c_H\$")
    savefig("figures/fwi$iter.png")
end

sess = Session(); init(sess)
BFGS!(sess, loss, vars = [E, cH], callback = cb)

# 
close("all")
# figure()
# lineview(sess, c_, loss, cH_true, run(sess, c_))
# figure()
# gradview(sess, c_, loss, run(sess, c_), scale=0.01)