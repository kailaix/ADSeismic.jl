using ADCME
using PyPlot 
matplotlib.use("agg")

n = 80
pml = 30
N = n + 2pml + 1
NT = 1500
Δt = 1.5/NT 
Δx = 1/n 
x0 = LinRange(0, 1, n+1)
xE = Array((0:n+2pml)*Δx .- pml*Δx)
xH = (xE[2:end]+xE[1:end-1])/2
C = 1000.0

## PML
σE = zeros(N)
for i = 1:pml
    d = i*Δx
    σE[pml + n + 1 + i] = C* (d/(pml*Δx))^3
    σE[pml+1-i] = C* (d/(pml*Δx))^3
end

σH = zeros(N-1)
for i = 1:pml 
    d = (i-1/2)*Δx 
    σH[pml + n + i] = C* (d/(pml*Δx))^3
    σH[pml+1-i] = C* (d/(pml*Δx))^3
end

## Source
function ricker(dt = 0.002, f0 = 3.0)
    nw = 2/(f0*dt)
    nc = floor(Int, nw/2)
    t = dt*collect(-nc:1:nc)
    b = (π*f0*t).^2
    w = @. (1 - 2b)*exp(-b)
    # w = @. exp(-b)
end
R = ricker() * 1e7
if length(R)<NT+1
    R = [R;zeros(NT+1-length(R))]
else
    R = R[1:NT+1]
end
R_ = constant(R)

idx_src = pml + Int(round(0.0/Δx))+1
Z = zeros(N)
Z[idx_src] = 1.0
Z = Z[2:end-1]

idx_rcv = [pml + Int(round(0.0/Δx))+2, pml + Int(round(1.0/Δx))+1]

## Anomaly
model = "step"
cH = ones(length(xH)) * 3.0
cH_init = copy(cH)
cH[pml + Int(round(0.6/Δx)):end] .= 1.0
# model = "slop"
# cH = ones(length(xH)) * 2.0
# cH_init = copy(cH)
# cH[pml + Int(round(0.3/Δx)):end] = range(2.0, 3.0, length=length(cH[pml + Int(round(0.3/Δx)):end]))


cH_true = copy(cH)
cE = (cH[1:end-1]+cH[2:end])/2

## Similation
function condition(i, E_arr, H_arr)
    i<=NT+1
end

function body(i, E_arr, H_arr)
    E = read(E_arr, i-1)
    H = read(H_arr, i-1)

    # second order
    # ΔH = cH * (E[2:end]-E[1:end-1])/Δx - σH*H
    # H += ΔH * Δt
    # ΔE = cE * (H[2:end]-H[1:end-1])/Δx - σE[2:end-1]*E[2:end-1] + R_[i] * Z
    # E = scatter_add(E, 2:N-1, ΔE * Δt)

    ## forth order
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

E = stack(E)
H = stack(H)

sess = Session(); init(sess)
E_true, H_true = run(sess, [E, H])

## data
figure(figsize = (14,4))
subplot(132)
pcolormesh(xE, (0:NT)*Δt, E_true, cmap="seismic", vmax=maximum(E_true), vmin=-maximum(E_true), rasterized=true)
plot(xE[idx_src]*ones(NT+1), (0:NT)*Δt, "r", label="Source", alpha=0.8)
plot(xE[idx_rcv[1]]*ones(NT+1), (0:NT)*Δt, "--", label="Receiver #1", alpha=0.8)
plot(xE[idx_rcv[2]]*ones(NT+1), (0:NT)*Δt, "--", label="Receiver #2", alpha=0.8)
autoscale(enable=true, axis="x", tight=true)
legend(loc="upper center")
xlabel("Localtion (km)")
ylabel("Time (s)")
subplot(133)
plot((0:NT)*Δt, E_true[:,idx_rcv[1]], label="Receiver #1", alpha=0.8)
plot((0:NT)*Δt, E_true[:,idx_rcv[2]], label="Receiver #2", alpha=0.8)
legend()
gca().ticklabel_format(axis="y", style="sci", scilimits=(0,0))
autoscale(enable=true, axis="x", tight=true)
xlabel("Time (s)")
ylabel("Amplitude")
subplot(131)
plot(xH, cH_true)
autoscale(enable=true, axis="x", tight=true)
xlabel("Location (km)")
ylabel("Velocity (km/s)")
tight_layout()
savefig("data.png", bbox_inch="tight")
savefig("data.pdf", bbox_inch="tight", dpi=300)