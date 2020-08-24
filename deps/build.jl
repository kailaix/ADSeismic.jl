using ADCME

cd("CustomOps")
if !isdir("build")
    mkdir("build")
end
cd("build")
ADCME.cmake()
ADCME.make()