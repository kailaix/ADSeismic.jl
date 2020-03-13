using ADCME

function install_binary(DIR)
    cd(DIR)
    rm("build", recursive=true, force=true)
    mkdir("build")
    cd("build")
    ADCME.cmake(DIR)
    ADCME.make()
end

install_binary("$(@__DIR__)/CustomOps/AcousticWaveSource")
install_binary("$(@__DIR__)/CustomOps/SourceOps")
install_binary("$(@__DIR__)/CustomOps/ReceiveOps")

