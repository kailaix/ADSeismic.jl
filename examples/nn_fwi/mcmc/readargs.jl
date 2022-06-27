using ArgParse


function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table! s begin
        "--scaling"
            default = 0.002
            arg_type = Float64
        "--nsim"
            default = 10
            arg_type = Int64
        "label"
            default = "nothing"
    end

    return parse_args(s)
end

parameters = parse_commandline()
println(parameters)

label = parameters["label"]
scaling = parameters["scaling"]
nsim = parameters["nsim"]