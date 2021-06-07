using Revise
using ADCME
using ADSeismic
using PyPlot
using DelimitedFiles

num_repeat = 3

open("acoustic-time-$(has_gpu() ? "gpu" : "cpu").txt", "a") do io 
  write(io, "===============================================\n")
end

for (k, scale) in enumerate(101:100:1001)

    param = AcousticPropagatorParams(NX=scale, NY=scale, NSTEP=100, DELTAT=1e-4,  DELTAX=1.0, DELTAY=1.0)

    rc = Gauss(param, 1/pi, 100, 1e6)
    srci = [div(param.NX,2)]
    srcj = [div(param.NY,2)]
    srcv = reshape(rc, :, 1)
    src = AcousticSource(srci, srcj, srcv)

    C = 3300*ones(param.NX+2, param.NY+2)
    model = AcousticPropagatorSolver(param, src, C)

    sess = Session(); init(sess)
    u = run(sess, model.u)

    time = 0
    for i = 1:num_repeat
      init(sess)
      time += (@timed run(sess, model.u))[2]
    end
    time /= num_repeat

    @info time, scale
    open("acoustic-time-$(has_gpu() ? "gpu" : "cpu").txt", "a") do io 
      writedlm(io, [scale*scale time])
    end

end



# figure(figsize=(4.5,3))
# scales = readdlm("acoustic-scales.txt")
# cpu = readdlm("acoustic-time_cpu.txt")
# gpu = readdlm("acoustic-time_gpu.txt")
# plot(scales, cpu, ".-", label="CPU")
# plot(scales, gpu, ".-", label="GPU")
# ticklabel_format(style="sci", axis="x", scilimits=(0,0))
# # xlabel("Degrees of freedom")
# xlabel("Nx × Ny")
# ylabel("Time (sec)")
# grid("on", which="both")
# legend()
# savefig("acoustic_benchmark.png", bbox_inches="tight")
# savefig("acoustic_benchmark.pdf", bbox_inches="tight")
