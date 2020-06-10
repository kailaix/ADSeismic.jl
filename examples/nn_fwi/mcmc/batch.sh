set +x 
export TF_FORCE_GPU_ALLOW_GROWTH=true
export CUDA_VISIBLE_DEVICES=0,1
smt configure -c store-diff
smt configure -d Data
smt configure --addlabel cmdline 
smt configure -e julia


for idx in 1 2 3
do  
for s in 0.0005 0.001 0.002 0.005 0.01
do 
smt run -m NNFWI-reg.jl --nsim 5000 --scaling $s &
sleep 1
done 
done
wait 