#!/bin/bash
#SBATCH --ntasks=21
#SBSTCH --cpus-per-task-8
##SBATCH --ntasks-per-node=8
###SBATCH --ntasks-per-socket=2
##SBATCH --gpus=4
##SBATCH --gpus-per-node=4
##SBATCH --cpus-per-gpu=4
##SBATCH --constraint=[GPU_SKU:V100_PCIE|GPU_SKU:V100S_PCIE|GPU_SKU:V100_SXM2]
##SBATCH --constraint=[GPU_MEM:32GB]
#SBATCH --partition=serc
#SBATCH --time=7-00:00:00
#SBATCH --output=./stdout/"%x_%j".out
##SBATCH --export=ALL
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wayne.weiqiang@gmail.com
#SBATCH -v

module -q purge
#module load julia
#module load cudnn
#module -q load cuda/10.0
module load openmpi/4.0.3
export MPI_C_LIBRARIES=/share/software/user/open/openmpi/4.0.3/lib/libmpi.so
export MPI_INCLUDE_PATH=/share/software/user/open/openmpi/4.0.3/include

mpirun -n 21 julia $SLURM_JOB_NAME

#echo "$(date): job $SLURM_JOBID starting on $SLURM_NODELIST"
#while true; do
#    echo "$(date): normal execution"
#    sleep 60
#done
