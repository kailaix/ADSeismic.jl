using Documenter, ADSeismic

makedocs(sitename="ADSeismic.jl", 
pages = Any["index.md",  
        "ad.md",
        "forward_simulation.md",
        "backward_inversion.md", 
        "contents.md", 
        "api.md"])


deploydocs(
    repo = "github.com/kailaix/ADSeismic.jl.git",
)
