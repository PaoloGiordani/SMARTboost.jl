
module SMARTboost

export SMARTloglikdivide, SMARTparam, SMARTdata, SMARTfit, SMARTpredict, SMARTcvfit, SMARTrelevance, SMARTpartialplot,
 SMARTmarginaleffect, SMARTpc, SMARTtree, SMARTboostTrees

using Distributed, SharedArrays, LinearAlgebra, Statistics, DataFrames,Dates, Random
import Optim, LineSearches, SpecialFunctions

include("param.jl")
include("loss.jl")
include("struct.jl")                  # options related to fitting and cv are explained here
include("fit.jl")
include("validation.jl")
include("main_functions.jl")          # collects main functions called by the user
include("PSoderlindPrintTable.jl")    # Paul Soderlind's function for nice output printing. Use PrettyTables instead?



end
