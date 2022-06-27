for scale in 201 301 401 501 601 701 801 901 1001
do 
    srun --gres=gpu:1 --partition=k80 julia acoustic_gpu_simulation_level.jl $scale & 
done 
wait 