mpirun -n 1 julia MPI_backward.jl &
mpirun -n 4 julia MPI_backward.jl &
wait