using Revise
using JLD2
using Test
using DelimitedFiles

@load "data/C.jld2" C VX
CC = C
VXX = VX
@load "data/C0.jld2" C VX
C0 = C
VX0 = VX
@load "data/C1.jld2" C VX
C1 = C
VX1 = VX
@load "data/C2.jld2" C VX
C2 = C
VX2 = VX
@load "data/C3.jld2" C VX
C3 = C
VX3 = VX
CMPI = [C0 C1;C2 C3]
VXMPI = [VX0 VX1; VX2 VX3]

writedlm(stdout, CC)
println("-------------------------------------------------------------")
writedlm(stdout, CMPI)
println("-------------------------------------------------------------")
writedlm(stdout, CMPI-CC)
println("-------------------------- Relative ----------------------------------")
writedlm(stdout, abs.(CMPI-CC) ./ (1e-10 .+abs.(CC)))