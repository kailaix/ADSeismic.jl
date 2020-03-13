using Revise
using ADCME
using ADSeismic
using PyPlot
using DelimitedFiles
use_gpu(0)

num_repeat = 3
scales = zeros(Int64, 6)
time = zeros(6)
for (k,scale) in enumerate(100:100:600)
    # param = AcousticPropagatorParams(NX=Int(round(scale*sqrt(2))), NY=Int(round(scale*sqrt(2))), NSTEP=100, DELTAT=1e-4,  DELTAX=1.0, DELTAY=1.0)
    param = AcousticPropagatorParams(NX=scale, NY=scale, NSTEP=100, DELTAT=1e-4,  DELTAX=1.0, DELTAY=1.0)
    # rc = Gauss(param, f0 = 200.0)
    a = 1/pi
    rc = Gauss(param, a, 100, 1e6)

    srci = [div(param.NX,2)]
    srcj = [div(param.NY,2)]
    srcv = reshape(rc, :, 1)
    src = AcousticSource(srci, srcj, srcv)

    C = 3300^2*ones(param.NX+2, param.NY+2)
    model = AcousticPropagatorSolver(param, src, C)

    sess = Session(); init(sess)
    u = run(sess, model.u)
    for i = 1:num_repeat
      time[k] += (@timed run(sess, model.u))[2]
    end
    time[k] /= num_repeat
    # scales[k] = Int(round(sqrt(scale)))*Int(round(sqrt(scale)))
    scales[k] = scale*scale
    @info time[k], scale
end
writedlm("acoustic-scales.txt", scales)
writedlm("acoustic-time_$(has_gpu() ? "gpu" : "cpu").txt", time)


figure(figsize=(4.5,3))
scales = readdlm("acoustic-scales.txt")
cpu = readdlm("acoustic-time_cpu.txt")
gpu = readdlm("acoustic-time_gpu.txt")
plot(scales, cpu, ".-", label="CPU")
plot(scales, gpu, ".-", label="GPU")
ticklabel_format(style="sci", axis="x", scilimits=(0,0))
# xlabel("Degrees of freedom")
xlabel("Nx × Ny")
ylabel("Time (sec)")
grid("on", which="both")
legend()
savefig("acoustic_benchmark.png", bbox_inches="tight")
savefig("acoustic_benchmark.pdf", bbox_inches="tight")
