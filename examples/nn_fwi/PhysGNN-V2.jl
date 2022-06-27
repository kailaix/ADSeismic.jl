using Revise
using ADSeismic
using ADCME
using MAT
using PyPlot
using PyCall
using DelimitedFiles
# matplotlib.use("Agg")
close("all")
if has_gpu()
  gpu = true
else
  gpu = false
end


data_dir = "data/acoustic"
if !ispath(data_dir)
  mkpath(data_dir)
end
figure_dir = "figure/PhysGNN_V2/"
if !ispath(figure_dir)
  mkpath(figure_dir)
end
result_dir = "result/PhysGNN_V2/"
if !ispath(result_dir)
  mkpath(result_dir)
end
# if isfile(joinpath(result_dir, "loss.txt"))
#   rm(joinpath(result_dir, "loss.txt"))
# end

################### Inversion using Automatic Differentiation #####################
reset_default_graph()
batch_size = 8 ## number of models per source
sample_size = 8 ## number of sampled y per source
model_name = "models/marmousi2-model-smooth.mat"

## load model
params = load_params(model_name)
src = load_acoustic_source(model_name)
rcv = load_acoustic_receiver(model_name)
nsrc = length(src)
rcv = Array{Any}(undef, nsrc, batch_size)
for i = 1:nsrc
  for j = 1:batch_size
    rcv[i,j] = load_acoustic_receiver(model_name)[i]
  end
end

vp0 = matread(model_name)["vp"]
mean_vp0 = mean(vp0)
std_vp0 = std(vp0)
vp0 = constant(vp0)

## load data
Rs = Array{Array{Float64,2}}(undef, length(src))
for i = 1:length(src)
    Rs[i] = readdlm(joinpath(data_dir, "marmousi-r$i.txt"))
end

## original fwi
# vp_ = Variable(vp0 / mean_vp0)
# vp = vp_ * mean_vp0
# vars = vp_

## add NN
x, isTrain, y, z = sampleUQ(batch_size, sample_size, (size(src[1].srcv,1)+1,length(rcv[1].rcvi)), 
                            z_size=8, base=4, ratio=size(vp0)[1]/size(vp0)[2]) 
x = x * std_vp0
vp = vp0 + tf.slice(x, (size(x).-(size(x)[1], size(vp0)...)).÷2, (size(x)[1], size(vp0)...))

## assemble acoustic propagator model
# model = x->AcousticPropagatorSolver(params, x, vp^2)
models = Array{Any}(undef, nsrc, batch_size)
for i = 1:nsrc
  for j = 1:batch_size
    vp = vp0 + tf.slice(x[j], (size(x[j]).-size(vp0)).÷2, size(vp0))
    models[i,j] = x->AcousticPropagatorSolver(params, x, vp)
  end
end

@pywith tf.device("/cpu:0") begin
  global variable_rcv_ = placeholder(Float64, shape=(nsrc, size(Rs[1])...))
  global Rs_ = Array{PyObject}(undef, nsrc, sample_size)
  for i = 1:nsrc
    for j = 1:sample_size
        Rs_[i,j] = variable_rcv_[i] + y[i, j]
    end
  end

  global variable_src = Array{Any}(undef, nsrc)
  global si_ = placeholder(Int32, shape=[nsrc,length(src[1].srci)])
  global sj_ = placeholder(Int32, shape=[nsrc,length(src[1].srcj)])
  global sv_ = placeholder(Float64, shape=[nsrc,size(src[1].srcv)...])
  for i = 1:nsrc
    variable_src[i] = AcousticSource(si_[i], sj_[i], sv_[i])
  end

  global loss = sampling_compute_loss_and_grads_GPU_v2(models, variable_src, rcv, Rs_, reg=200.0, method="lp")
end

lr = 0.001
opt = AdamOptimizer(lr).minimize(loss, colocate_gradients_with_ops=true)

si = zeros(size(si_)...)
sj = zeros(size(sj_)...)
sv = zeros(size(sv_)...)
for i = 1:nsrc
  si[i,:] = src[i].srci
  sj[i,:] = src[i].srcj
  sv[i,:,:] = src[i].srcv
end
variable_rcv = zeros(size(variable_rcv_)...)
for i = 1:nsrc
  variable_rcv[i,:,:] = Rs[i]
end

dic = Dict(
  isTrain=>true,
  z=>rand(Float32, size(z)...),
  y=>randn(size(y)...),
  si_=>si,
  sj_=>sj,
  sv_=>sv,
  variable_rcv_=>variable_rcv
)

sess = Session(); init(sess)
@info "Initial loss: ", run(sess, loss, feed_dict=dic)


## run inversion
losses = []
σ = 0.01
fixed_z = randn(Float32, size(z)...)
time = 0
for iter = 1:1000
    if iter%10==1
      dic = Dict(
        isTrain=>true,
        z=>fixed_z
      )
      plot_result(sess, x, dic, iter, dirs=figure_dir, var_name="x")
      plot_result(sess, vp, dic, iter, dirs=figure_dir, var_name="vp")
    end

    dic = Dict(
      isTrain=>true,
      z=>rand(Float32, size(z)...),
      y=>randn(size(y)...) .* σ .* [std(Rs[i]) for i = 1:nsrc],
      si_=>si,
      sj_=>sj,
      sv_=>sv,
      variable_rcv_=>variable_rcv
    )
    
    global time += @elapsed ls, _ = run(sess, [loss, opt], feed_dict=dic)
    if iter == 1
      global mean_ls = ls
    else
      global mean_ls
      mean_ls += 1/(iter+1) * (ls - mean_ls)
    end
    # println("[#$iter] loss = $loss_, mean loss = $mean_ls")
    println("   $iter\t$ls\t$mean_ls")
    println(" * time: $time")
    push!(losses, ls)
end