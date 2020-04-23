using ADCME

cd("CustomOps")
rm("build", force=true, recursive=true)
mkdir("build")
cd("build")
ADCME.cmake()
ADCME.make()