NVCC_COMPILER = readlines(pipeline(`which nvcc`))[1]
s = join(readlines(pipeline(`nvcc --version`)), " ")
ver = match(r"V(\d+\.\d)", s)[1]
if ver[1:2]!="10"
    error("TensorFlow backend of ADCME requires CUDA 10.0. But you have CUDA $ver")
end
if ver[1:4]!="10.0"
    @warn("TensorFlow is compiled using CUDA 10.0, but you have CUDA $ver. This might cause some problems.")
end

pkg_dir = "$(homedir())/.julia/conda/3/pkgs/"
files = readdir(pkg_dir)
libpath = filter(x->startswith(x, "cudatoolkit") && isdir(joinpath(pkg_dir,x)), files)
if length(libpath)==0
    @warn "cudatoolkit* not found in $pkg_dir"
elseif length(libpath)>1
    @warn "more than 1 cudatoolkit found, use $(libpath[1]) by default"
end

if length(libpath)>=1
    LIBCUDA = abspath(joinpath(pkg_dir, libpath[1], "lib"))
end


libcudatoolkit_path = filter(x->startswith(x, "cudnn") && isdir(joinpath(pkg_dir,x)), files)
if length(libcudatoolkit_path)==0
    @warn "cudnn* not found in $pkg_dir"
elseif length(libcudatoolkit_path)>1
    @warn "more than 1 cudatoolkit found, use $(libpath[1]) by default"
end

if length(libcudatoolkit_path)>=1
    LIBCUDA = LIBCUDA*":"*abspath(joinpath(pkg_dir, libcudatoolkit_path[1], "lib"))
    @info " --------------- CUDA include headers  --------------- "
    cudnn = joinpath(pkg_dir, libcudatoolkit_path[1], "include", "cudnn.h")
    cp(cudnn, joinpath(TF_INC, "cudnn.h"), force=true)
end

CUDA_INC = joinpath(splitdir(splitdir(NVCC)[1])[1], "include")
