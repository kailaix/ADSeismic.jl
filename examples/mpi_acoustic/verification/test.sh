mpirun -n 1 julia MPI_forward.jl &
mpirun -n 4 julia MPI_forward.jl &
wait 

