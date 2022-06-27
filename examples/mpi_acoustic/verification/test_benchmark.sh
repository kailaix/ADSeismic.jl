ulimit -u 1000000
for n in 25 16 9 4 1 
do 
mpirun -n $n julia MPI_forward_benchmark.jl &
done
wait
