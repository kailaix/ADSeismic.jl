# using Revise
using ADCME
using ADSeismic
using PyPlot
using DelimitedFiles
include("database.jl")
scale = 201

if length(ARGS)==1
    scale = parse(Int64, ARGS[1])
end

num_repeat = 3


param = AcousticPropagatorParams(NX=scale, NY=scale, 
    NSTEP=100, DELTAT=1e-4,  DELTAX=1.0, DELTAY=1.0,
    PropagatorKernel = 1)

rc = Gauss(param, 1/pi, 100, 1e6)
srci = [div(param.NX,2)]
srcj = [div(param.NY,2)]
srcv = reshape(rc, :, 1)
src = AcousticSource(srci, srcj, srcv)

C = placeholder(3300*ones(param.NX+2, param.NY+2))
model = AcousticPropagatorSolver(param, src, C)

loss = sum(model.u^2)
g = gradients(loss, C)
sess = Session(); init(sess)
u = run(sess, model.u)
_ = run(sess, g)

time = 0
for i = 1:num_repeat
    init(sess)
    global time += (@timed run(sess, model.u))[2]
end
time /= num_repeat
t1 = time 

time = 0
for i = 1:num_repeat
    init(sess)
    global time += (@timed run(sess, g))[2]
end
time /= num_repeat
t2 = time  

insert_record("cpu_simulation", scale, t1, t2)