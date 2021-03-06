ulimit -u 50000
for scale in 201 301 401 501 601 701 801 901 1001
do 
    julia acoustic_cpu_tensor_level.jl $scale & 
    julia acoustic_cpu_simulation_level.jl $scale & 
done 
wait 