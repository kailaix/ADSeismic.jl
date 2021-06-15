using Revise
using ADCME
using ADSeismic
using PyPlot
using DelimitedFiles

num_repeat = 3

open("elastic-time-$(has_gpu() ? "gpu" : "cpu").txt", "a") do io 
    write(io, "===============================================\n")
end
  
for (k, scale) in enumerate(101:100:1001)

    param = ElasticPropagatorParams(NX=scale, NY=scale, NSTEP=100, DELTAT=1e-4,  DELTAX=1.0, DELTAY=1.0) 
    
    source = Gauss(param, 1/pi, 100, 1e6) + 1
    srci = [div(param.NX,2)]
    srcj = [div(param.NY,2)]
    srctype = [0]
    srcv = reshape(source, :, 1)
    src = ElasticSource(srci, srcj, srctype, srcv)

    vp = 3300.
    vs = 3300. / 1.732
    rho = 2800.
    λ, ρ, μ = compute_lame_parameters(param.NX, param.NY, vp, vs, rho)
    model = ElasticPropagatorSolver(param, src, ρ, λ, μ)

    sess = Session(); init(sess)    
    vs = run(sess, model.vx)

    time = 0
    for i = 1:num_repeat
      init(sess)
      time += (@timed run(sess, model.vx))[2]
    end
    time /= num_repeat

    @info time, scale
    open("elastic-time-$(has_gpu() ? "gpu" : "cpu").txt", "a") do io 
        writedlm(io, [scale*scale time])
    end

end
