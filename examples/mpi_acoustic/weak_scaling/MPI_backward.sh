for n in 1 4 9 16 25 36 49 64 81 100 
do 
    salloc -n $n --ntasks-per-node=1 mpirun -n $n julia MPI_backward.jl &
done 
wait