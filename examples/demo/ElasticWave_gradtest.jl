# ENV["CUDA_VISIBLE_DEVICES"] = ""
using Revise
using ADCME
using ADSeismic
using PyPlot
using DelimitedFiles
matplotlib.use("agg")
o = placeholder(ones(1))
param = ElasticPropagatorParams(NX=150, NY=150, NSTEP=5, DELTAT=1e-4, DELTAX=1.0, DELTAY=1.0, 
                                vp_ref = 3300.0, USE_PML_XMIN=true, USE_PML_XMAX=true, USE_PML_YMIN=true, USE_PML_YMAX=true) 

source = Ricker(param, 15.0, 100.0, 1e6) 
srci = repeat([param.NX ÷ 2], 5)
srcj = repeat([param.NY ÷ 2], 5)
srctype = [0;1;2;3;4] # vx, vy, sigmaxx, sigmayy, sigmaxy
srcv = reshape(source, :, 1)
src = ElasticSource(srci, srcj, srctype, srcv)


vp = 3000. * ones(param.NX+2, param.NY+2)
vs = vp / 1.732
rho = 2800. * ones(param.NX+2, param.NY+2)
vp0 = vp
vp_ = Variable(vp[:]) + o[1]^2
vp = reshape(vp_, (152,152))
λ, ρ, μ = compute_lame_parameters(param.NX, param.NY, vp, vs, rho)
model = ElasticPropagatorSolver(param, src, ρ , λ, μ)

sess = Session(); init(sess)    


# loss = sum(model.vx)
loss = sum(model.vx^2) + sum(model.vy^2) + sum(model.sigmaxx^2) + sum(model.sigmaxy^2) + sum(model.sigmayy^2) 
grad = gradients(loss, o)

# vx = run(sess, model.vx)
# p = visualize_wavefield(vx, param)
# saveanim(p, "elastic-wavefield.gif")

function test(x)
  
  l = run(sess, loss, o=>x)
  g = run(sess, grad, o=>x)
  
  return l, g

end

close("all")
ADCME.test_gradients(test, ones(1))
savefig("gradtest_sigmaxy_gpu.png")
