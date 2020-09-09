using PyPlot 
using DelimitedFiles

weak = readdlm("timing.txt")

figure()
pm = sortperm(weak[:,1])
weak = weak[pm,:]
semilogx(weak[:,1], weak[:,2], "o-", label = "Forward")
semilogx(weak[:,1], weak[:,3], "o-", label = "Backward")
xlabel("Number of Processors", fontsize = 20)
ylabel("Time (sec)", fontsize = 20)
gca().xaxis.set_tick_params(labelsize=20)
gca().yaxis.set_tick_params(labelsize=20)
legend(fontsize=18)
grid("on")
tight_layout()
savefig("elastic_weak.png")