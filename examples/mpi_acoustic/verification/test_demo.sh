mpirun -n 1 julia MPI_forward_demo.jl &
mpirun -n 4 julia MPI_forward_demo.jl &
wait 

