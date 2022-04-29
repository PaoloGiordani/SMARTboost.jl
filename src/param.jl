
#
# SMARTparam    struct. Includes several options for cross-validation.
# SMARTdata     struct
# SMARTparam    function
# SMARTdata     function
#
# convert_df_matrix   # converts a dataframe to a Matrix{Float64} of x
#

# NOTE: several fields are redundant, reflecting early experimentation, and will disappear in later versions
mutable struct SMARTparam{T<:AbstractFloat, I<:Int}

    loss::Symbol             # loss function (log-likelihood)
    coeff::Vector{T}         # coefficients (if any) used in computing the loss function
    verbose ::Symbol         # :On, :Off
    # options for cross-validation (apply to ntrees and the outer loop over depth or other paramter)
    randomizecv::Bool        # true to scramble data for cv, false for contiguous blocks (PG: safer when there is a time-series aspect)
    nfold::I                 # n in n-fold cv. 1 for a single validation set (early stopping in the case of ntrees)
    sharevalidation::Union{T,I}   # e.g. 0.2 or 1230. Relevant if nfoldcv = 1. The LAST block of observations is used if randomizecv = false
    stderulestop::T         # e.g. 0.05. Applies to stopping while adding trees to the ensemble
    stopwhenlossup::Bool    # in cv the outer loop over depth or other parameter (other than ntrees), stops as soon as loss increass if true

    # learning rate
    lambda::T

    # Tree structure and priors
    depth::I
    sigmoid::Symbol  # which simoid function. :sigmoidsqrt or :sigmoidlogistic. sqrt x/sqrt(1+x^2) 10 times faster than exp.
    meanlnτ::T              # Assume a student-t for log(tau).
    varlnτ::T               #
    doflnτ::T

    varμ::T                # Helps in preventing very large mu, which otherwise can happen in final trees. 1.0 or even 2.0
    dofμ::T

    # sub-sampling and pre-selection of features
    subsamplesharevs::T    # if <1.0, only a random share of the sample is used in variable selection and in refinement of (mu,tau).
                           # All observations then used to estimate beta|i,mu,tau, unless the next option is true.
    subsamplefinalbeta::Bool   # Relevant only if subsamplesharevs < 1.0. If true, it then uses the sub-sample even in estimating beta|i,mu,tau
    subsampleshare_columns::T  # if <1.0, only a random share of features is used at each split (re-drawn at each split)

    # grid and optimization parameters
    μgridpoints::I  # points at which to evaluate μ during variable selection. 5 is sufficient on simulated data, but actual data can benefit from more (due to with highly non-Gaussian (and non-uniform))
    τgridpoints::I  # points at which to evaluate τ during variable selection. 1-5 are supported. If less than 3, refinement is then on a grid with more points

    refineOptimGrid::Bool         # if false, refinement (after feature selection) uses optim() and parallelization is over 7 values of tau. If true, uses a grid(mu,tau), twice, and parallelizes so that
                                 # each core handles only ONE evaluation. Up to 24 workers. I thought this would be much faster when n is large, but I was wrong: it is slower even with n = 10_000_000, 32 cores.
                                 # Perhaps this is due to the need to copy G etc... (large matrices) on each core? Perhaps @Thread would work better+
    xtolOptim::T                 # tolerance in the optimization, only if RefineGrid=false of μ. e.g. 0.01 (measured in dμ). Higher numbers are less precise but faster.
    optimizevs::Bool            # true to run a full optimization over μ (not τ) in the variable selection stages. Slows down the code and should not be necessary except perhaps in extremely non-linear functions
                                # ? eliminate ?

    sharptree::Bool        # false for smooth trees (τ estimated), false for sharpe sigmoid functions. NB: extremely inefficient implementation! Use CatBoost instead.

    # others

    ntrees::I   # number of trees
    R2p::T   # If 0.899, then 0.899 is used for the first tree, and then it is updated to the fit of the first tree. Can have a sizable impact for small n and low signal-to-noise
    p0::I
    loglikdivide::T  # the log-likelhood is divided by this scalar. Used to improve inference when observations are correlated.
    overlap::I       # used in purged-CV

end

struct SMARTdata{T<:AbstractFloat,D<:Any}
    y::Vector{T}
    x::Matrix{T}
    dates::Vector{D}
    fdgp::Vector{T}   # fdgp (n) vector of true E(y|x), for simulations with known data-generating-process
    fnames::Vector{String}
end


"""
    SMARTparam(;)
Parameters for SMARTboost

# Inputs that are more likely to be modified by user (all inputs are keywords with default values)
- `loss:Symbol`             [:L2] currently only :L2 is supported, extensions coming soon.
- `lambda::Float`           [0.20] learning rate. 0.2 is a good compromise. Consider 0.1 for best performance.
- `depth::Int`              [4] tree depth. If not default, then typically cross-validated in SMARTcvfit.
- `nfold::Int`              [5] n in n-fold cv. Set nfold = 1 for a single validation set, the last sharevalidation share of the sample.
- `sharevalidation:`        [0.30] Size of the validation set (integer), last sharevalidation rows of data (then forces nfold=1). Or float, share of validation set (then relevant only if nfold = 1 (see nfold))
- `loglikdivide::Float`     [1.0] with panel data, SMARTloglikdivide() can be used to set this parameter
- `overlap::Int`            [0] number of overlaps. Typically overlap = h-1, where y(t) = Y(t+h)-Y(t). Use for purged-CV.
- `verbose::Symbol`         [:Off] verbosity :On or :Off

# Inputs that may sometimes be modified by user (all inputs are keyword with default values)
- `T::Type`                 [Float32] Float32 is faster than Float64
- `ntrees::Int`             [2000] Important for SMARTbst, which always run to ntrees. SMARTfit can stop before ntrees using cv.
- `randomizecv::Bool`       [false] default is purged-cv (see paper); a time series or panel structure is automatically detected (see SMARTdata)
- `coeff::Vector`           [] additional coefficient needed to compute loss function; not relevant for L2 loss
- `sigmoid::Symbol`         [:sigmoidsqrt] :sigmoidlogistic is more familiar but slower than :sigmoidlogistic
- `subsamplesharevs::Float` [1.0] row subs-sampling; if <1.0, only a randomly drawn (at each iteration) share of the sample is used in determining ι (which feature),μ,τ.
- `subsamplefinalbeta`      [1.0] relevant only if subsamplesharevs < 1.0. If true, uses the sub-sample even in estimating beta|i,mu,tau, else the entire sample.
- `subsampleshare_columns`  [1.0] column sub-sampling.
- `stderulestop::Float`     [0.01] A positive number stops iterations while the loss is still decreasing. This results in faster computations at minimal loss of fit.

# Inputs that should not require changing unless suspecting highly nonlinear functions (all inputs are keyword with default values)
- `μgridpoints::Int`        [10] number of points at which to evaluate μ during variable selection. 5 is sufficient on simulated data, but actual data may benefit from more (due to with highly non-Gaussian features
- `τgridpoints::Int`        [3]  number of points at which to evaluate τ during variable selection. 1-5 are supported.
- `xtolOptim::Float`        [0.02] tolerance in the optimization. 0.02 seems sufficiently low
- `optimizevs::Bool`        [false] true to run a full optimization over μ (not τ) in the variable selection stages
- `meanlnτ::Float`          [0.0] prior mean of log(τ). 0.0 is quasi-linear for Gaussian features (and sigmoid for highly leptokurtic features)
- `varlnτ::Float`           [1.0] prior dispersion of log(τ). Lower numbers encourage quasi-linearity more strongly.
                            Note: this is actually a dispersion, and it is divided by param.depth.
- `doflnτ::Float`           [5.0] degrees of freedom for prior student-t distribution of log(τ)
- `varμ::Float`             [4.0] tighter numbers encourage splits closer to the mean of each feature
- `dofμ::Float`             [10.0]

# Note: type (Float32 (default) or Float64) is set in SMARTdata().

# Example
    param = SMARTparam()
# Example
    param = SMARTparam(nfold=1,sharevalidation=0.2)
# Example
    lld,ess =  SMARTloglikdivide(df,:excessret,:date,overlap=h-1)
    param   =  SMARTparam(data,loglikdivide=lld,overlap=h-1)
"""
function SMARTparam(;

    T::Type = Float32,   # Float32 or Float64. Float32 is up to twice as fast for sufficiently large n (due not to faster computations, but faster copying and storing)
    loss = :L2,            # loss function
    coeff = coeff_loss(loss),      # coefficients (if any) used in loss
    verbose = :Off,      # level of verbosity, :On, :Off
    randomizecv = false,        # true to scramble data for cv, false for contiguous blocks ('Block CV')
    nfold       = 5,                 # n in n-fold cv. 1 for a single validation set (early stopping in the case of ntrees)
    sharevalidation = 0.30,      #[0.30] Size of the validation set (integer), last sharevalidation rows of data. Or float, share of validation set (then relevant only if nfold = 1 (see nfold))
    stderulestop = 0.01,         # e.g. 0.05. Applies to stopping while adding trees to the ensemble. larger numbers give smaller ensembles.
    stopwhenlossup = false,    # in cv the outer loop over depth or other parameter (other than ntrees), stops as soon as loss increass if true
    lambda = 0.20,
    # Tree structure and priors
    depth  = 4,        # 3 allows 2nd degree interaction and is fast. 4 takes almost twice as much on average. 5 can be 8-10 times slower.
    sigmoid = :sigmoidsqrt,  # :sigmoidsqrt or :sigmoidlogistic
    meanlnτ= 0.0,    # Assume a Gaussian for log(tau). See CalibratingTauPrior.jl.
    varlnτ = 1.0,  # centers toward quasi-linearity. This is the (DISPERSION) paametes of the student-t distribution (not the variance ).
    doflnτ = 5.0,    # NB: varlnτ dispersion is divided by param.depth
    varμ   = 2.0^2,    # smaller number make it increasingly unlikely to have nonlinear behavior in the tails. DISPERSION, not variance
    dofμ   = 10.0,
    # sub-sampling and pre-selection of features
    subsamplesharevs = 1.0,    # if <1.0, only a random share of the sample is used in variable selection and in refinement of (mu,tau).
                                # All observations then used to estimate beta|i,mu,tau, unless the next option is true. ! Works poorly in my simulations !
    subsamplefinalbeta = false,   # Relevant only if subsamplesharevs < 1.0. If true, it then uses the sub-sample even in estimating beta|i,mu,tau
    subsampleshare_columns = 1.0,  # if <1.0, only a random share of features is used at each split (re-drawn at each split)

    # grid and optimization parameters
    μgridpoints = 10,  # points at which to evaluate μ during variable selection. 5 is sufficient on simulated data, but actual data can benefit from more (due to with highly non-Gaussian features
    τgridpoints = 3,  # points at which to evaluate τ during variable selection. 1-5 are supported. If less than 3, refinement is then on a grid with more points

    refineOptimGrid = false, # if false, refinement (after feature selection) uses optim() and parallelization is over 7 values of tau. If true, uses a grid(mu,tau), twice, and parallelizes so that
                                 # each core handles only ONE evaluation. Up to 24 workers. I thought this would be much faster when n is large, but I was wrong: it is slower even with n = 10_000_000, 32 cores.
    xtolOptim = 0.02,  # tolerance in the optimization, only if RefineGrid=false of μ. e.g. 0.01 (measured in dμ). Higher numbers are less precise but faster.
    optimizevs = false, # true to run a full optimization over μ (not τ) in the variable selection stages. Should not be necessary except with extremely non-linear functions

    # sharptree is currently implemented with a default of tau = 50.0. Much larger numbers can cause problems, because the optimizer can try mu such that one side of the
    # split is empty. sharptree may require larger μgridpoints (>= 10), since tau is not there to add flexibility.
    sharptree = false,        # false for smooth trees (τ estimated), false for sharpe sigmoid functions

    # miscel

    ntrees = 2000,   # number of trees
    R2p = 0.898,   # If 0.898, then 0.898 is used for the first tree, and then it is updated to the fit of the first tree. Can have a sizable impact for small n and low signal-to-noise
    p0 = 1,
    loglikdivide = 1.0,   # the log-likelhood is divided by this scalar. Used to improve inference when observations are correlated.
    overlap = 0)

    I = typeof(1)
    
    @assert(doflnτ>T(2), " doflnτ must be greater than 2.0 (for variance to be defined) ")
    @assert(T(0) <= R2p < T(0.99), " R2p must be between 0.0 and 0.99 ")
    @assert(T(1e-20) < xtolOptim, "xtolOptim must be positive ")
    if depth>I(5); @warn "Depth larger than five is slow, particularly with few workers, and occasionally unstable (NaN outcomes). Is it really needed? Is CV loss decreasing with depth all the way to $(depth-1)?"; end

    # The following works even if sharevalidation is a Float which is meant as an integer (e.g. in R wrapper)
    if T(sharevalidation)==T(round(sharevalidation))  # integer
        sharevalidation=I(sharevalidation)
        nfold = I(1)
    else
        sharevalidation = T(sharevalidation)
    end

    param = SMARTparam( loss,T.(coeff),Symbol(verbose),randomizecv,I(nfold),T(sharevalidation),T(stderulestop),stopwhenlossup,T(lambda),I(depth),Symbol(sigmoid),
        T(meanlnτ),T(varlnτ),T(doflnτ),T(varμ),T(dofμ),T(subsamplesharevs),subsamplefinalbeta,T(subsampleshare_columns),I(μgridpoints),I(τgridpoints),refineOptimGrid,T(xtolOptim),optimizevs,sharptree,I(ntrees),T(R2p),I(p0),T(loglikdivide),I(overlap) )
    
    return param
end


# converts a dataframe to a Matrix{Float64} of x
function convert_df_matrix(df,T)
  n,p = size(df)
  x   = Matrix{T}(undef,n,p)
  for i in 1:p
    x[:,i] = T.(df[:,i])
  end
  return x
end


"""
        SMARTdata(y,x,param,[dates];T=Float32,fdgp=y,fnames=nothing,enforce_dates=true)
Collects data in preparation for fitting SMARTboost

# Inputs

- `y::Vector`:              Vector of responses.
- `x`:                      Matrix of features. Can be a vector or matrix of floats, or a dataframe. Converted internally to a Matrix{T}, T as defined in SMARTparam
- `param::SMARTparam`:
- `dates::AbstractVector`:  (keyword) [1:n] Typically Vector{Date} or Vector{Int}. Used in cross-validation to determine splits.
                            If not supplied, the default 1:n assumes a cross-section of independent realizations (conditional on x) or a single time series.
- `T`:                      [Float32] Type to which all y and x is converted. In default, all data is converted to Float32 (faster).
- `fdgp::Vector{T}`:        (keyword) true E(y|x), for simulations with known data-generating-process
- `fnames::Vector{String}`: (keyword) [feature1, feature2, ... ] feature names
- `enforce_dates::Bool`:    (keyword) [true] enforces dates to be chronological, i.e. sorted(unique(dates)) = unique(dates)

# Examples of use
    data = SMARTdata(y,x,param)
    data = SMARTdata(y,df[:,[:CAPE, :momentum ]],param)
    data = SMARTdata(y,df[:,3:end],param)
    data = SMARTdata(y,x,param,dates=df.dates,fnames=df.names)
    s = std(data.y)
    lastdate = maximum(data.dates)

# Notes
- y and x are converted to type T, where T is defined in SMARTparam as either Float32 or Float64.
- When dates are provided, they should, as a rule, be in chronological order in the sense that sort(dates)==dates (for cross-validation functions)
"""
function SMARTdata(y::AbstractVector,x::Union{AbstractVector,AbstractMatrix,AbstractDataFrame},param::SMARTparam,dates::AbstractVector = Vector{Int64}[];fnames = Vector{String}[], fdgp = y, enforce_dates::Bool = true)  # NB: works

    T    = typeof(param.lambda)
    length(fnames)==0 ? fnames = ["feature $i" for i in 1:size(x,2)] : fnames = fnames

    if length(dates)==0 || enforce_dates==false
        dates = collect(1:length(y))
    else
        datesu  = unique(dates)
        @assert(datesu==sort(datesu), " Dates must be in chronological order: unique(dates) is required to be in ascending order (for cross-validation functions). ")
    end

    if typeof(x)<:AbstractDataFrame
        data = SMARTdata(T.(y),convert_df_matrix(x,T),dates,T.(fdgp),fnames)
    elseif typeof(x)<:AbstractVector
        data = SMARTdata(T.(y),convert(Matrix{T},reshape(x,length(x),1)),dates,T.(fdgp),fnames)
    elseif typeof(x)<:AbstractMatrix
        data = SMARTdata(T.(y),convert(Matrix{T},x),dates,T.(fdgp),fnames)
    end

    return data
end
