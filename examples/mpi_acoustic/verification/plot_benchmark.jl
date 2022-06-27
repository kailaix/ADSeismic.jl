# sh test_benchmark.sh
using PyPlot
using DelimitedFiles

r = readdlm("data/benchmark.txt")
idx = sortperm(r[:,1])
r = r[idx,:]
figure(figsize=(10,4))
subplot(121)
bar(1:size(r,1), r[:,2], tick_label = Int64.(r[:,1]))
xlabel("Number of Processors")
ylabel("Time (seconds)")

subplot(122)
plot(r[:,1], r[1,2]./r[:,2], label="Speedup")
legend()
xlabel("Number of Processors")

savefig("benchmark.png")