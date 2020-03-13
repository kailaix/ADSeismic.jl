using Revise
using ADCME
using ADSeismic
using PyPlot
using DelimitedFiles
use_gpu()

num_repeat = 3
scales = zeros(Int64, 6)
time = zeros(6)
for (k,scale) in enumerate(100:100:600)
    # param = ElasticPropagatorParams(NX=Int(round(scale*sqrt(2))), NY=Int(round(scale*sqrt(2))), NSTEP=100, DELTAT=1e-4,  DELTAX=1.0, DELTAY=1.0)
    # param = ElasticPropagatorParams(NX=Int(round(sqrt(scale))), NY=Int(round(sqrt(scale))), NSTEP=100, DELTAT=1e-4,  DELTAX=1.0, DELTAY=1.0)
    param = ElasticPropagatorParams(NX=scale, NY=scale, NSTEP=100, DELTAT=1e-4,  DELTAX=1.0, DELTAY=1.0) 
    a = 1/pi
    rc = Gauss(param, a, 100, 1e6)

    srci = [div(param.NX,2)]
    srcj = [div(param.NY,2)]
    srctype = [0]
    srcv = reshape(rc, :, 1)
    src = ElasticSource(srci, srcj, srctype, srcv)

    vp = 3300.
    vs = 3300. / 1.732
    rho = 2800.
    λ, ρ, μ = compute_default_properties(param.NX, param.NY, vp, vs, rho)

    model = ElasticPropagatorSolver(param, src, ρ, λ, μ)

    sess = Session(); init(sess)
    vx = run(sess, model.vx)
    # time[k] = (@timed run(sess, model.vx))[2]
    for i = 1:num_repeat
      time[k] += (@timed run(sess, model.vx))[2]
    end
    time[k] /= num_repeat
    # scales[k] = Int(round(sqrt(scale)))*Int(round(sqrt(scale)))
    scales[k] = scale*scale
    @info time[k], scale
end
writedlm("elastic-scales.txt", scales)
writedlm("elastic-time_$(has_gpu() ? "gpu" : "cpu").txt", time)


figure(figsize=(4.5,3))
scales = readdlm("elastic-scales.txt")
cpu = readdlm("elastic-time_cpu.txt")
gpu = readdlm("elastic-time_gpu.txt")
plot(scales, cpu, ".-", label="CPU")
plot(scales, gpu, ".-", label="GPU")
# xlabel("Degrees of freedom")
ticklabel_format(style="sci", axis="x", scilimits=(0,0))
xlabel("Nx × Ny")
ylabel("Time (sec)")
grid("on", which="both")
legend()
savefig("elastic_benchmark.png", bbox_inches="tight")
savefig("elastic_benchmark.pdf", bbox_inches="tight")