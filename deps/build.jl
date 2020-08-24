using ADCME

PWD = pwd()
cd("CustomOps")
if !isdir("build")
    mkdir("build")
end
cd("build")
ADCME.cmake()
ADCME.make()
cd(PWD)