# Getting Started

**An Open Source High Performance Package for General Seismic Inversion Problems**

![](./assets/icon.png)

ADSeismic is suitable for general inversion problems in seismic imaging. The packages implements the forward simulation of acoustic and elastic wave fields and allows inversion of physical properties such as media densities, Lamé parameters, shear modulus, etc. by means of **automatic differentiation**. For example, the following problems fall into the scope of this framework

- Full waveform inversion (FWI)
- Rupture inversion
- Source-time inversion

The package provides a unified interface for both acoustic and elastic wave simulation. Users only need to specify the geometries and known parameters of the physical simulation. Gradients (traditionally derived by adjoint methods) are computed automatically. Some notable features of this package are

- **Battery included**: unified approach to various seismic inversion problems such as full waveform inversion (FWI), rupture inversion and source-time inversion.
- **High performance**: computational graph optimization and parallel computing. 
- **(Multi-)GPU support**: support for GPU acceleration as well as multi-GPU (in theory TPUs are also supported).
- **Easy-to-use**: no adjoint state method codes required for building your own inversion models; automatic differentiation is the workhorce.

ADSeismic is built on [ADCME.jl](https://github.com/kailaix/ADCME.jl/), an automatic differentiation library for computational mathematics and engineering. The former is fully compatible with ADCME.jl, which indicates that this package (ADSeismic.jl) can serve as a acoustic/elastic simulation solver in a system inverse modeling problem. 

## Installation

`ADCME` is a dependency for using this package. Install `ADCME` with
```julia
using Pkg; Pkg.add("ADCME")
```

Then install the lastest ADSeismic.jl by
```julia
using Pkg; Pkg.add("https://github.com/kailaix/ADSeismic.jl#master")
```
or the stable version by 
```julia
using Pkg; Pkg.add("ADCME")
```

If you want to use GPU for ADSeismic.jl, you should enable GPU support in ADCME by 
```julia
using ADCME
enable_gpu()
```
ADCME.jl looks for `nvcc` so be sure the CUDA compiler has been installed. 

After enabling GPU, recompile the ADSeismic.jl library with 
```julia
using Pkg; Pkg.build("ADSeismic.jl")
```



