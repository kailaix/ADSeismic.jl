using ADCME
using PyCall
using LinearAlgebra
using PyPlot
using Random
Random.seed!(233)

function acoustic_one_step(w,wold,phi,psi,sigma,tau,c,dt,hx,hy,nx,ny)
    acoustic_one_step_ = load_op_and_grad("./build/libAcousticOneStep","acoustic_one_step", multiple=true)
    w,wold,phi,psi,sigma,tau,c,dt,hx,hy,nx,ny = convert_to_tensor(Any[w,wold,phi,psi,sigma,tau,c,dt,hx,hy,nx,ny], [Float64,Float64,Float64,Float64,Float64,Float64,Float64,Float64,Float64,Float64,Int64,Int64])
    acoustic_one_step_(w,wold,phi,psi,sigma,tau,c,dt,hx,hy,nx,ny)
end

# TODO: specify your input parameters
use_gpu()
nx = 10
ny = 10
mask = zeros(nx+2, ny+2)
mask[2:end-1, 2:end-1] .= 1.0
mask = mask[:]

using Random; Random.seed!(1234)
hx = constant(0.1)
hy = constant(0.1)
dt = constant(0.1)
c = rand((nx+2)*(ny+2)) .* mask
phi = rand((nx+2)*(ny+2)) .* mask
psi = rand((nx+2)*(ny+2)) .* mask
wold = rand((nx+2)*(ny+2)) .* mask
w = rand((nx+2)*(ny+2)) .* mask
sigma = rand((nx+2)*(ny+2)) .* mask
tau = rand((nx+2)*(ny+2)) .* mask
u = acoustic_one_step(w,wold,phi,psi,sigma,tau,c,dt,hx,hy,nx,ny)
sess = Session(); init(sess)
@show run(sess, u)

# uncomment it for testing gradients
# error() 


# TODO: change your test parameter to `m`
#       in the case of `multiple=true`, you also need to specify which component you are testings
# gradient check -- v
function scalar_function(x)
    return sum(sum(acoustic_one_step(x,wold,phi,psi,sigma,tau,c,dt,hx,hy,nx,ny).^2))
end

# psi, w

# TODO: change `m_` and `v_` to appropriate values
m_ = constant(ones((nx+2)*(ny+2)) .* mask)
v_ = rand((nx+2)*(ny+2)) .* mask
y_ = scalar_function(m_)
dy_ = gradients(y_, m_)
ms_ = Array{Any}(undef, 5)
ys_ = Array{Any}(undef, 5)
s_ = Array{Any}(undef, 5)
w_ = Array{Any}(undef, 5)
gs_ =  @. 1000 / 10^(1:5)

for i = 1:5
    g_ = gs_[i]
    ms_[i] = m_ + g_*v_
    ys_[i] = scalar_function(ms_[i])
    s_[i] = ys_[i] - y_
    w_[i] = s_[i] - g_*sum(v_.*dy_)
end

sess = Session(); init(sess)
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
savefig("gradtest")