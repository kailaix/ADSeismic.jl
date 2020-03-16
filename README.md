
![](./docs/src/assets/icon.png)
# ADSeismic: An Open Source High Performance Package for General Seismic Inversion Problems


| Documentation                                                | Build Status                                                 |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [![](https://img.shields.io/badge/docs-dev-blue.svg)](https://kailaix.github.io/ADSeismic.jl/dev/) | [![Build Status](https://travis-ci.com/kailaix/ADSeismic.jl.svg?branch=master)](https://travis-ci.com/kailaix/ADSeismic.jl) |

ADSeismic is suitable for general inversion problems in seismic imaging. The packages implements the forward simulation of acoustic and elastic wave fields and allows inversion of physical properties such as media densities, Lamé parameters, shear modulus, etc., by means of **automatic differentiation**. Several features of this package is

- **Battery included**: unified approach to various seismic inversion problems such as full waveform inversion (FWI), rupture inversion and source-time inversion.
- **High performance**: computational graph optimization and parallel computing. 
- **(Multi-)GPU support**: support for GPU acceleration as well as multi-GPU (in theory TPUs are also supported).
- **Easy-to-use**: no adjoint state method codes required for building your own inversion models; automatic differentiation is the workhorce.

See our paper [A General Approach to Seismic Inversion with Automatic Differentiation](https://arxiv.org/pdf/2003.06027.pdf) for more details. 

## Installation

ADSeismic.jl requires proper installation of [ADCME.jl](https://github.com/kailaix/ADCME.jl/).
```julia
using Pkg
Pkg.add("ADCME")
```
This might take up to 20 minutes for complete installation. 

Then install this package (ADSeismic.jl) by 
```julia
using Pkg
Pkg.add("ADSeismic")
```

To enable GPU support, make sure `nvcc` is available on your machine, and then
```julia
using ADCME
enable_gpu()
```



