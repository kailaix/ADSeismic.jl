using ADCME 
using PyPlot 

db = Database("simulation.db")
c = execute(db, "select size, forward, backward from acoustic where level = 'cpu_tensor' and size <= 600 order by size")
c = collect(c)
s = [x[1] for x in c]
t1 = [x[2] for x in c]
t2 = [x[3] for x in c]
close("all")
loglog(s, t1, "*-", label = "Forward Computation")
loglog(s, t2, "*-", label = "Gradient Backpropagation")
savefig("result.png")