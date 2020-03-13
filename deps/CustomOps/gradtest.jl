using ADCME
using PyCall
using LinearAlgebra
using PyPlot
using Random
Random.seed!(233)

################## Load Operator ##################
elastic_wave_op = load_op("build/libElasticWaveOp", "elastic_wave_op")

################## End Load Operator ##################

# TODO: specify your input parameters
NX = 10
NY = 10
lambda = constant(ones(NX+2, NY+2))
mu = constant(ones(NX+2, NY+2))
rho = constant(ones(NX+2, NY+2))
sigmaxx = constant(ones(NX+2, NY+2))
sigmayy = constant(ones(NX+2, NY+2))
sigmaxy = constant(ones(NX+2, NY+2))
vx = constant(ones(NX+2, NY+2))
vy = constant(ones(NX+2, NY+2))
mem = constant(ones(8, NX+2, NY+2))
ax = constant(ones(2,NX))
bx = constant(ones(2,NX))
kx = constant(ones(2,NX))
ay = constant(ones(2,NY))
by = constant(ones(2,NY))
ky = constant(ones(2,NY))
alphax = constant(ones(2,NX))
alphay = constant(ones(2,NY))
dx = constant(1.0)
dy = constant(1.0)
dt = constant(1.0)
srci = constant([1;2;3])
srcj = constant([4;5;6])
srctype = constant([1;1;1])
srcv = constant(rand(3))
# TODO: specify your input parameters
res = elastic_wave_op(lambda,mu,rho,sigmaxx,sigmayy,sigmaxy,vx,vy,mem,ax,bx,kx,alphax,ay,by,ky,alphay,dx,dy,dt,srci,srcj,srctype,srcv)
sess = Session()
init(sess)
run(sess, res)
error()

# TODO: change your test parameter to `m`
# gradient check -- v
function scalar_function(m)
    return sum(tanh(elastic_wave_op(lambda,mu,rho,sigmaxx,sigmayy,sigmaxy,vx,vy,mem,ax,bx,kx,alphax,ay,by,ky,alphay,dx,dy,dt,srci,srcj,srctype,srcv)))
end

# TODO: change `m_` and `v_` to appropriate values
m_ = constant(rand(10,20))
v_ = rand(10,20)
y_ = scalar_function(m_)
dy_ = gradients(y_, m_)
ms_ = Array{Any}(undef, 5)
ys_ = Array{Any}(undef, 5)
s_ = Array{Any}(undef, 5)
w_ = Array{Any}(undef, 5)
gs_ =  @. 1 / 10^(1:5)

for i = 1:5
    g_ = gs_[i]
    ms_[i] = m_ + g_*v_
    ys_[i] = scalar_function(ms_[i])
    s_[i] = ys_[i] - y_
    w_[i] = s_[i] - g_*sum(v_.*dy_)
end

sess = Session()
init(sess)
sval_ = run(sess, s_)
wval_ = run(sess, w_)
close("all")
loglog(gs_, abs.(sval_), "*-", label="finite difference")
loglog(gs_, abs.(wval_), "+-", label="automatic differentiation")
loglog(gs_, gs_.^2 * 0.5*abs(wval_[1])/gs_[1]^2, "--",label="\$\\mathcal{O}(\\gamma^2)\$")
loglog(gs_, gs_ * 0.5*abs(sval_[1])/gs_[1], "--",label="\$\\mathcal{O}(\\gamma)\$")

plt.gca().invert_xaxis()
legend()
xlabel("\$\\gamma\$")
ylabel("Error")
