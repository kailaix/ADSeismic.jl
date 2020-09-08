for n in 1 4 16 25 64 100
do 
    salloc -n $n -c 16 mpirun -n $n julia MPI_forward.jl &
done 
wait