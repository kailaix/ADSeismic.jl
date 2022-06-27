# salloc -n 1 -c 32 --partition=largemem mpirun -n 1 julia MPI_backward.jl &
# salloc -n 4 -c 12 --partition=largemem mpirun -n 4 julia MPI_backward.jl &

for n in 9 16 25 36 49 64 81 100 
do 
    salloc -n $n -c 32 mpirun -n $n julia MPI_backward.jl &
done 
wait