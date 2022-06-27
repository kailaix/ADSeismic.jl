for n in 1 4 9 16 25 36 49 64 81 100 
do 
    salloc -n $n -c 16 mpirun -n $n julia MPI_forward.jl &
done 
wait