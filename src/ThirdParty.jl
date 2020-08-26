export install_geodynamics
import ADCME: require_gfortran, require_file, GFORTRAN, http_file, change_directory
function install_geodynamics()
    PWD = pwd()
    require_gfortran()
    GEODYNAMICS = joinpath(ADCME.PREFIXDIR, "geodynamics")
    change_directory(GEODYNAMICS)
    require_file("seismic_CPML_2D_pressure_second_order.f90") do 
        http_file("https://raw.githubusercontent.com/kailaix/seismic_cpml/master/seismic_CPML_2D_pressure_second_order.f90", "seismic_CPML_2D_pressure_second_order.f90")
    end
    require_file("seismic_CPML_2D_isotropic_fourth_order.f90") do 
        http_file("https://raw.githubusercontent.com/kailaix/seismic_cpml/master/seismic_CPML_2D_isotropic_fourth_order.f90", "seismic_CPML_2D_isotropic_fourth_order.f90")
    end
    # compiling
    run(`$GFORTRAN -std=gnu -fimplicit-none -frange-check -O3 -fmax-errors=10 -pedantic -pedantic-errors -Waliasing -Wampersand -Wcharacter-truncation -Wline-truncation -Wsurprising -Wno-tabs -Wunderflow
            -mcmodel=medium seismic_CPML_2D_pressure_second_order.f90  -o acoustic`)
    run(`$GFORTRAN -std=gnu -fimplicit-none -frange-check -O3 -fmax-errors=10 -pedantic -pedantic-errors -Waliasing -Wampersand -Wcharacter-truncation -Wline-truncation -Wsurprising -Wno-tabs -Wunderflow
            -mcmodel=medium seismic_CPML_2D_isotropic_fourth_order.f90 -o elastic`)
    exec1 = joinpath(pwd(), "acoustic"), joinpath(pwd(), "elastic")
    cd(PWD)
    exec1
end