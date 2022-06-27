salloc -n 1 -c 32 --partition=largemem mpirun -n 1 julia MPI_backward.jl &
for n in 4 16 25 36 64 100 
do 
    salloc -n $n -c 32 mpirun -n $n julia MPI_backward.jl &
done 
wait