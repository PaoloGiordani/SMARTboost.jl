#
# SMARTtree           struct
# SMARTboostTrees     struct
# SMARTboostTrees     function
#

# information on one tree
# PG note: SVectors would make no difference speed-wise, since not building and storing this information barely changes timing (except possibly at VERY low n): not repeated enough times to make a difference

"""
    struct SMARTtree{T<:AbstractFloat,I<:Int}
Collects information about a single SMART tree of depth d

# Fields
- `i`:             vector, d indices of splitting features.
- `μ`:             vector, d values of μ at each split
- `τ`:             vector, d values of τ at each split
- `β`:             vector of dimension p (number of features), leaf values
- `fi2`:           vector, d values of feature importance squared: increase in R2 at each split.
"""
struct SMARTtree{T<:AbstractFloat,I<:Int}
    i::AbstractVector{I}                           # selected feature
    μ::AbstractVector{T}
    τ::AbstractVector{T}
    β::AbstractVector{T}
    fi2::AbstractVector{T}    # feature importance squared: increase in R2 at each split, a (depth,1) vector.
end

# information on the collection of trees
"""
    struct SMARTboostTrees{T<:AbstractFloat,I<:Int}
Collects information about the ensemble of SMART trees.

# Fields
- `param::SMARTparam`:        values of param
- `gamma0`:                   initialization (prior to fitting any tree) of natural parameter, e.g. mean(data.y) for regression
- `trees::Vector{SMARTtree}`: element i collects the info about the i-th tree
- `infeatures::Vector{Bool}`: element i is true if feature i is included in at least one tree
- `fi2::Vector`:              feature importance square
- `meanx`:                    vector of values m used in standardizing (x .- m')./s'
- `stdx`:                     vector of values m used in standardizing (x .- m')./s'
- `gammafit`:                 fitted values of natural parameter (fitted values of y for regression)
- `R2simul`
"""
mutable struct SMARTboostTrees{T<:AbstractFloat,I<:Int}
    param::SMARTparam
    gamma0::T
    trees::Vector{SMARTtree{T,I}}
    infeatures::Vector{Bool}
    fi2::Vector{T}          # feature importance squared: Breiman et al. 1984, equation 10.42 in Hastie et al, "The Elements of Statistical Learning", second edition
    meanx::Array{T}
    stdx::Array{T}
    gammafit::Vector{T}
    R2simul::Vector{T}
end


function SMARTboostTrees(param,gamma0,n,p,meanx,stdx)
    T  = typeof(param.varμ)
    trees, infeatures,R2simul,fi = SMARTtree{typeof(gamma0),typeof(p)}[], fill(false,p), T[], zeros(T,p)
    gammafit                     = fill(gamma0,n)
    SMARTtrees = SMARTboostTrees(param,gamma0,trees,infeatures,fi,meanx,stdx,gammafit,R2simul)
    return SMARTtrees
end
