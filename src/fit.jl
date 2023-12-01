
#
# Auxiliary functions called in the boosting algorithm
#
#  updateG
#  updateG_allocated
#  gridvectorτ
#  gridmatrixμ
#  sigmoidf
#  logpdft
#  lnpτ
#  lnpμ
#  fitβ                 MAP for Gaussian likelihood
#  Gfitβ
#  Gfitβ2
#  updateinfeatures
#  add_depth          add one layer to the tree, for given i, using a rough grid search (optional: full optimization)
#  loopfeatures       add_depth looped over features, to select best-fitting feature
#  refineOptim        having chosen a feature (typically via a rough grid), refines the optimization
#  preparedataSMART
#  preparegridsSMART
#  fit_one_tree
# updateSMARTtrees!
#  SMARTtreebuild
#  calibrate_R2total
#  set_lambda

function updateG(G0::AbstractArray{T},g::AbstractVector{T})::AbstractMatrix{T} where T<:AbstractFloat

   if length(size(G0)) == 1
         p, n = 1, length(G0)
    else
         n,p = size(G0)
    end

    G   = Matrix{T}(undef,n,p*2)

    @views for i in 1:p
        @. G[:,i]   =  G0[:,i]*g
        @. G[:,i+p] =  G0[:,i]*(1 - g)   #  =  G0[:,i] - G[:,i] is not faster
    end

    return G
end



function updateG_allocated(G0::AbstractArray{T},g::AbstractVector{T},G::AbstractMatrix{T})::AbstractMatrix{T} where T<:AbstractFloat

    p = size(G0,2)

    @views for i in 1:p
        @. G[:,i]   =  G0[:,i]*g
        @. G[:,i+p] =  G0[:,i]*(1 - g)   #  =  G0[:,i] - G[:,i] is not faster
    end

    return G
end


#Returns the grid of points at which to evaluate τ in the variable selection phase.
function gridvectorτ(meanlnτ::T,varlnτ::T,τgridpoints::Int;sharptree::Bool = false)::AbstractVector{T} where T<:AbstractFloat

    @assert(1 ≤ τgridpoints ≤ 5, "τgridpoints must be between 3 and 5")
    if sharptree
        τgrid = [100.0]
    else
        if τgridpoints==1
            τgrid = [5.0]
        elseif τgridpoints==2
            τgrid = [1.0, 9.0]
        elseif τgridpoints==3
            τgrid = [1.0, 3.0, 9.0]
        elseif τgridpoints==4
            τgrid = [1.0, 3.0, 9.0, 27.0]
        elseif τgridpoints==5
            τgrid = [1.0, 3.0, 9.0,27.0,50.0]
        end
    end

    return T.(τgrid)
end

#Returns a (npoints,p) matrix of values of μ at which to evaluate the sigmoidal function in the variable selection stage.
#I considered two ways of doing this: i) quantiles(unique( )), ii) clustering, like k-means or fuzzy k-means on each column of x.
#quantiles(unique()) rather than just quantiles() should work better if x[:,i] is sparse.
#At most maxn (default 100_000) observations are sampled.
#Note: If the matrix is sparse, e.g. lots of 0.0, but not dichotomous, in general k-means will NOT place a value at exactly zero (but close)
function gridmatrixμ(x::AbstractMatrix{T},npoints::Int; tol = 0.005, maxiter::Int = 100, fuzzy::Bool = false, maxn::Int = 100_000 ) where T<:AbstractFloat

    n,p = size(x)
    @assert(npoints<n,"npoints cannot be larger than n")
    mgrid       = SharedArray{T}(npoints,p)   # [loss, τ, μ]
    dichotomous = SharedVector{Bool}(p)

    n>maxn ? ssi=randperm(n)[1:maxn] : ssi=collect(1:n)

    @sync @distributed for i = 1:p
        dichotomous[i] = length(unique(x[:,i]))==2
        if dichotomous[i] == false
            mgrid[:,i] = quantile(x[:,i],[i for i = 1/(npoints+1):1/(npoints+1):1-1/(npoints+1)])
            u = unique(mgrid[:,i])
            lu = length(u)
            if lu<=3  #
                mgrid[1:lu,i] = u[1:lu]
                mgrid[lu+1:end,i] = quantile(unique(x[:,i]),[i for i = 1/(npoints+1-lu):1/(npoints+1-lu):1-1/(npoints+1-lu)])
            end

            #=
            # NOTE: Clustering.jl is not currently a dependency of SMARTboost
            if fuzzy==true
                r   = Clustering.fuzzy_cmeans(x[ssi,i]', npoints, 2.0; tol = tol, maxiter=maxiter, display = :none)
            else
                r   = Clustering.kmeans(x[ssi,i]', npoints; tol = tol, maxiter=maxiter, display = :none)  # input matrix is the adjoint (p,n). Output is Matrix
            end
            mgrid[:,i] = sort(vec(r.centers))
            =#
        end
    end

    return Array{T}(mgrid),Array{Bool}(dichotomous)    # (npoints,p) matrix of candidates for m
end


function sigmoidf(x::AbstractVector{T}, μ::T, τ::T,sigmoid::Symbol;dichotomous::Bool = false) where T<:AbstractFloat

    if dichotomous   # return 0 if x<=0 and x>1 otherwise. x is assumed de-meaned
        g = @. T(0.0) + T(1.0)*(x>0.0)
        return g
    end

    if sigmoid==:sigmoidsqrt
         g = @. T(0.5) + T(0.5)*( T(0.5)*τ*(x-μ)/sqrt(( T(1.0) + ( T(0.5)*τ*(x-μ) )^2  )) )
    elseif sigmoid==:sigmoidlogistic
        g = @. T(1.0) - T(1.0)/(T(1.0) + (exp(τ * (x - μ))))
    end

    return g
end



function logpdft(x::T,m::T,s::T,v::T) where T<:AbstractFloat
z       = ( x - m)/s
logpdfz = T(-0.5723649429247001)+SpecialFunctions.loggamma((v+T(1))/T(2))-SpecialFunctions.loggamma(v/T(2))-T(0.5)*log(v)-T(0.5)*(T(1)+v)*log(T(1)+(z^2)/v)
return logpdfz - log(s)
end

function lnpμ(μ::Union{T,Vector{T}},varμ::T,dofμ::T) where T<:AbstractFloat
#    s  = sqrt(varμ*(dofμ-2.0)/dofμ)  # to intrepret varlnτ as an actual variance. But prior more intuitive in terms of dispersion.
    s  = sqrt(varμ)
    lnp = sum(logpdft.(μ,T(0),s,dofμ))
    return T(lnp)
end

function lnpτ(τ::Union{T,Vector{T}},meanlnτ::T,varlnτ::T,doflnτ::T,depth) where T<:AbstractFloat # depth = param.depth, same at all levels
#    s  = sqrt((varlnτ/)*(doflnτ-2.0)/doflnτ)  # to intrepret varlnτ as an actual variance. But prior more intuitive in terms of dispersion.
    s  = sqrt(varlnτ/depth)
    lnp = sum(logpdft.(log.(τ),meanlnτ,s,doflnτ))
    return T(lnp)
end

#=
MAP for Gaussian density with Gaussian, non-conjugate prior for β and conditioning on given varϵ.

Notes
- Two steps taken to eliminate the occurrence of SingularException errors
  (These are very rare, but can happen if depth > 4 and the same feature is present more than once):
  1) If a leaf is near-empty, the diagonal of GG is increased very slightly (essentially slightly stronger prior on near empty leaves)
  2) try ... catch of SingularException, with progressively stronger prior (Pb)
  Eventually, for depth too large, these will greatly slow down the code. Depth > 5 not recommended with current code.
=#
function fitβ(r::AbstractVector{T},h::AbstractVector{T},G::AbstractArray{T},param::SMARTparam,varϵ::T,infeaturesfit,dichotomous,μ::Union{T,Vector{T}},τ::Union{T,Vector{T}},dichotomous_i::Bool)::Tuple{T,Vector{T},Vector{T}}  where T<:AbstractFloat

    var_r   = T(varϵ)/(T(1.0)-param.R2p)  # varϵ is computed on the pseudo-residuals
    n,p     = size(G)

    GGh=G'G

    # There are three steps taken to reduce the risk of SingularExpection errors: 1) diagonal of Pb is not exactly pb*I. I don't really understand why this helps with Float32, but I see no harm in it.
    # 2) If a leaf is near-empty, the diagonal of GGh is increased very slightly. 3) try ... catch, and increase the diagonal of Pb if needed.
    # try beta = ... catch err if i
    # To reduce the danger of near-singularity associated with empty leaves, bound elements on the diagonal of GGh, here at 0.001% of the largest element.
    # NOTE: Still needed?
    [GGh[i,i] = maximum([GGh[i,i],T(maximum(diag(GGh))*0.00001)])  for i in 1:p]

    # NOTE: the addition of [ T(0.0001)*i for in in 1:p ] in at least one case eliminate the problem with singular exception in a Float32 case
    # Consider eliminating it once I understand the problem better.
    Pb    =   (sum(diag(GGh))/(n*var_r*param.R2p))*I(p)

    # handling  possible SingularExceptions. These are very rare, but can happen if depth > 4 and the same feature is present more than once
    β           = zeros(T,p)  # seems needed to avoid the "beta not defined ... on worker 2"
    try
        β           = (GGh + varϵ*param.loglikdivide*Pb )\(G'r)
    catch err
        if isa(err,SingularException)
           #@warn "near-singularity detected: increasing prior tighteness gradually until invertibility is achieved."
           while isa(err,SingularException)
              try
                  err         = "no error"
                  Pb          = Pb*2.01    # 2.01 rather than T(2.01) on purpose: switch to Float64 if there are invertibility problems.
                  β           = (GGh + varϵ*param.loglikdivide*Pb )\(G'r)
              catch err
              end
           end
       end
    end

    Gβ   = G*β

    loglik  = -T(0.5)*( (sum((r .- Gβ).^2)/varϵ) )/param.loglikdivide  # loglik  = -T(0.5)*( n*T(log(2π)) + n*log(varϵ) + (sum((r .- Gβ).^2)/varϵ) )/param.loglikdivide

    if dichotomous_i
        logpdfμ, logpdfτ = T(0.0), T(0.0)
    elseif param.sharptree==true
        logpdfτ = T(0.0)
        logpdfμ = lnpμ(μ,param.varμ,param.dofμ)
    else
        logpdfμ = lnpμ(μ,param.varμ,param.dofμ)
        logpdfτ = T(lnpτ(τ,param.meanlnτ,param.varlnτ,param.doflnτ,param.depth))
    end

    loss  = -( loglik + logpdfτ + logpdfμ)

    return T(loss),T.(Gβ),T.(β)

end



function Gfitβ(r::AbstractVector{T},h::AbstractVector{T},G0::AbstractArray{T},xi::AbstractVector{T},param::SMARTparam,varϵ::T,infeaturesfit,dichotomous,μlogτ,dichotomous_i::Bool,G::AbstractMatrix{T})::T where T<:AbstractFloat

    μ = μlogτ[1]
    τ = exp(μlogτ[2])
    τ = maximum((τ,T(0.2)))  # Anything lower than 0.2 is still essentially linear, with very flat log-likelihood

    gL  = sigmoidf(xi,μ,τ,param.sigmoid,dichotomous=dichotomous_i)
    G   = updateG_allocated(G0,gL,G)

    loss, yfit,β = fitβ(r,h,G,param,varϵ,infeaturesfit,dichotomous,μ,τ,dichotomous_i)

    return loss
end



function Gfitβ2(r::AbstractVector{T},h::AbstractVector{T},G0::AbstractArray{T},xi::AbstractVector{T},param::SMARTparam,varϵ::T,infeaturesfit,dichotomous,μv,τ::T,dichotomous_i::Bool,G::AbstractMatrix{T})::T where T<:AbstractFloat

    μ = T(μv[1])
    τ = maximum((τ,T(0.2)))  # Anything lower than 0.2 is still essentially linear, with very flat log-likelihood

    gL  = sigmoidf(xi,μ,τ,param.sigmoid,dichotomous=dichotomous_i)
    G   = updateG_allocated(G0,gL,G)

    loss, yfit,β = fitβ(r,h,G,param,varϵ,infeaturesfit,dichotomous,μ,τ,dichotomous_i)

    return loss
end


# keeps track of which feature have been selected at any node in any tree (0-1)
function updateinfeatures(infeatures,ifit)
    x = deepcopy(infeatures)
    for i in ifit
        x[i] = true
    end
    return x
end

# For a given feature, adds one layer to a symmetric tree
# Selects a feature using a rough grid in τ|μ and assuming that the loss if monotonic in τ|μ and then refines the optimization for the selected feature.
# If loss increases, break loop over tau (reduces computation costs by some 25%)
function add_depth(t)

    T = typeof(t.varϵ)
    lossmatrix = fill(T(Inf64),length(t.τgrid),length(t.μgridi))

    n,p = size(t.G0)
    G   = Matrix{T}(undef,n,2*p)

    if t.dichotomous_i==true   # no optimization needed
        loss = Gfitβ(t.r,t.h,t.G0,t.xi,t.param,t.varϵ,t.infeaturesfit,t.dichotomous,[T(0.0),T(0.0)],t.dichotomous_i,G)
        τ,μ  = T(999.9), T(0.0)
    else
        for (indexμ,μ) in enumerate(t.μgridi)
            for (indexτ,τ) in enumerate(t.τgrid)
                lossmatrix[indexτ,indexμ] = Gfitβ(t.r,t.h,t.G0,t.xi,t.param,t.varϵ,t.infeaturesfit,t.dichotomous,[μ,log(τ)],t.dichotomous_i,G)
                if indexτ>1 && (lossmatrix[indexτ,indexμ])>(lossmatrix[indexτ-1,indexμ]); break; end  #  if loss increases, break loop over tau (reduces computation costs by some 25%)
            end
        end

        minindex = argmin(lossmatrix)  # returns a Cartesian index
        loss     = lossmatrix[minindex]
        τ        = t.τgrid[minindex[1]]
        μ        = t.μgridi[minindex[2]]

        # Optionally, further optimize over μ. Perhaps needed for highly nonlinear functions.
        if t.param.optimizevs==true
            μ0   = [μ]
            res  = Optim.optimize( μ -> Gfitβ2(t.r,t.h,t.G0,t.xi,t.param,t.varϵ,t.infeaturesfit,t.dichotomous,μ,τ,t.dichotomous_i,G),μ0,Optim.BFGS(linesearch = LineSearches.BackTracking()), Optim.Options(iterations = 100,x_tol = t.param.xtolOptim))
            loss = res.minimum
            μ    = res.minimizer[1]
        end
    end

    return [loss,τ, μ]
end




function loopfeatures(r::AbstractVector{T},h::AbstractVector{T},G0::AbstractArray{T},x::SharedMatrix{T},ifit,infeatures,μgrid::AbstractArray{T},dichotomous,τgrid::AbstractVector{T},param::SMARTparam,
    varϵ::T)::AbstractArray{T} where T<:AbstractFloat

    p           = size(x,2)
    outputarray = SharedArray{T}(p,3)   # [loss, τ, μ]
    outputarray[:,1] = fill(T(Inf),p)

    ps = [i for i in 1:p]

    if param.subsampleshare_columns < T(1.0)
        psmall = convert(Int64,round(p*param.subsampleshare_columns))
        ps     = ps[randperm(p)[1:psmall]]                  # subs-sample, no reimmission
    end

    @sync @distributed for i in ps
        t   = (r=r,h=h,G0=G0,xi=x[:,i],infeaturesfit=updateinfeatures(infeatures,i),dichotomous=dichotomous,μgridi=μgrid[:,i],dichotomous_i=dichotomous[i],τgrid=τgrid,param=param,varϵ=varϵ)
        outputarray[i,:] = add_depth(t)     # [loss, τ, μ]
    end

    return Array(outputarray)   # convert SharedArray to Array
end



# After completing the first step (selecting a feature), use μ0 and τ0 as starting points for a more refined optimization. Uses Optim
function refineOptim(r::AbstractVector{T},h::AbstractVector{T},G0::AbstractArray{T},xi::AbstractVector{T},infeaturesfit::Vector{Bool},dichotomous::Vector{Bool},μ0::T,dichotomous_i::Bool,τ0::T,
    param::SMARTparam,varϵ::T) where T<:AbstractFloat

    if dichotomous_i
        gL  = sigmoidf(xi,μ0,τ0,param.sigmoid,dichotomous=dichotomous_i)
        n,p = size(G0)
        G   = Matrix{T}(updateG_allocated(G0,gL,Matrix{T}(undef,n,p*2)))
        loss,τ,μ = T(Inf),τ0,μ0
    else

        # optimize tau on a grid, and mu by BFGS. If param.μgridpoints is smaller, the grid is wider
        if param.sharptree==true
            τgrid = T[τ0]
        elseif param.τgridpoints == 1
            τgrid = convert(Vector{T},exp.([ log(τ0)+j for j = -2.7:0.3:2.7 ]))
        elseif param.τgridpoints == 2
            τgrid = convert(Vector{T},exp.([ log(τ0)+j for j = -1.8:0.3:1.8 ]))
        else
            if τ0<8.0
                τgrid = convert(Vector{T},exp.([ log(τ0)+j for j = -0.9:0.3:0.9 ]))
            else
                τgrid = convert(Vector{T},exp.([ log(τ0)+j for j = -0.9:0.3:1.8 ]))   # allow max tau=50
            end
        end

        lossmatrix = SharedArray{T}(length(τgrid),2)
        lossmatrix = fill!(lossmatrix,T(Inf))

        @sync @distributed for indexτ = 1:length(τgrid)
            res = optimize_mutau(r,h,G0,xi,param,varϵ,infeaturesfit,dichotomous,τgrid[indexτ],dichotomous_i,μ0,T)
            lossmatrix[indexτ,1] = res.minimum
            lossmatrix[indexτ,2] = res.minimizer[1]
        end

        lossmatrix = Array{T}(lossmatrix)

        minindex = argmin(lossmatrix[:,1])
        loss  = lossmatrix[minindex,1]
        τ     = τgrid[minindex]
        μ     = lossmatrix[minindex,2]

    end

    return loss,τ,μ
end

# G  created here
function optimize_mutau(r,h,G0,xi,param,varϵ,infeaturesfit,dichotomous,τ,dichotomous_i,μ0,T)
    n,p = size(G0)
    G   = Matrix{T}(undef,n,p*2)
    res  = Optim.optimize( μ -> Gfitβ2(r,h,G0,xi,param,varϵ,infeaturesfit,dichotomous,μ,τ,dichotomous_i,G),[μ0],
    Optim.BFGS(linesearch = LineSearches.BackTracking()), Optim.Options(iterations = 100,x_tol = param.xtolOptim/T(1+(τ>10))  ))
end


# Preliminary operations on data before starting boosting loop: standardize x using robust measures of dispersion.
# In the first version of the paper and code, I computed the robust std only on non-zero values, now I computed on all values,
# which implies a prior of sharper, less smooth functions on features with lots of zeros.
function preparedataSMART(data,param)

    T     = typeof(param.varμ)
    meanx = T.(mean(data.x,dims=1))  # changed to mean for continuous features
    stdxL2 = std(data.x,dims=1)    # this alone is very poor with sparse data in combinations with default priors on mu and tau (x/stdx becomes very large)
    stdx  = copy(stdxL2)

    for i in 1:size(data.x)[2]
        if length(unique(data.x[:,i]))>2     # with 0-1 data, stdx = 0 when computed on non-zero data.
            meanx[i] = T(median(data.x[:,i]))  # median rather than mean
            xi        = data.x[:,i]
            #xi        = xi[abs.(xi) .> T(0.0001*stdx[i])]  # Old versions: selects non-zero values
            s_median   = 1.42*median( abs.( xi .- meanx[i]) ) # NOTE: use of robust measure of std, not std, and computed only on non-zero values
            if s_median==T(0)
                stdx[i]=stdxL2[i]
            else
                stdx[i]=minimum(vcat(s_median,stdxL2[i]))
            end
        end
    end

    data_standardized = SMARTdata(data.y,(data.x .- meanx)./stdx,param,data.dates)

    return data_standardized,meanx,stdx
end

# data.x is now assumed already standardized
function preparegridsSMART(data,param)
    T    = typeof(param.varμ)
    τgrid             = gridvectorτ(param.meanlnτ,param.varlnτ,param.τgridpoints,sharptree=param.sharptree)
    μgrid,dichotomous = gridmatrixμ(data.x,param.μgridpoints)
    n,p               = size(data.x)
    return τgrid,μgrid,dichotomous,n,p
end


function fit_one_tree(r::AbstractVector{T},h::AbstractVector{T},x::AbstractArray{T},infeatures,μgrid,dichotomous,τgrid,param::SMARTparam) where T<:AbstractFloat

    var_wr = var(r)
    varϵ   = var_wr*(T(1.0) - param.R2p)

    n, p   = size(x)
    G0     = ones(T,n,1)    # initialize G, the matrix of features
    loss0  = T(Inf)

    yfit0, ifit,μfit,τfit,infeaturesfit,fi2,βfit = zeros(T,n),Int64[],T[],T[],copy(infeatures),zeros(T,param.depth),T[]
    subsamplesize  = convert(Int64,round(n*param.subsamplesharevs))

    if param.subsamplesharevs == 1.0
        ssi         = collect(1:n)
    else
        #ssi         = sample([i for i in 1:n],subsamplesize,replace = false)    # sub-sample indexes. Sub-sample no reimmission
        ssi         = randperm(n)[1:subsamplesize]  # subs-sample, no reimmission
    end

    for depth in 1:param.depth     #  NB must extend G for this to be viable

        # variable selection
        if param.subsamplesharevs == 1.0
            outputarray = loopfeatures(r,h,G0,x,ifit,infeaturesfit,μgrid,dichotomous,τgrid,param,varϵ)  # loops over all variables
        else            # Variable selection using a random sub-set of the sample. All the sample is then used in refinement.
            if length(h)==1
                outputarray = loopfeatures(r[ssi],h,G0[ssi,:],SharedMatrix(x[ssi,:]),ifit,infeaturesfit,μgrid,dichotomous,τgrid,param,varϵ)  # loops over all variables
            else
                outputarray = loopfeatures(r[ssi],h[ssi],G0[ssi,:],SharedMatrix(x[ssi,:]),ifit,infeaturesfit,μgrid,dichotomous,τgrid,param,varϵ)  # loops over all variables
            end
        end

        i          = argmin(outputarray[:,1])  # outputarray[:,1] is loss (minus log marginal likelihood) vector
        τ0, μ0     = outputarray[i,2], outputarray[i,3]

        infeaturesfit = updateinfeatures(infeaturesfit,i)

        # refine optimization, after variable selection
        if param.subsamplesharevs<T(1.0) && param.subsamplefinalbeta==true
            if length(h)==1
                loss,τ,μ = refineOptim(r[ssi],h,G0[ssi,:],SharedVector(x[ssi,i]),infeaturesfit,dichotomous,μ0,dichotomous[i],τ0,param,varϵ)
            else
                loss,τ,μ = refineOptim(r[ssi],h[ssi],G0[ssi,:],SharedVector(x[ssi,i]),infeaturesfit,dichotomous,μ0,dichotomous[i],τ0,param,varϵ)
            end
        else
            loss,τ,μ = refineOptim(r,h,G0,SharedVector(x[ssi,i]),infeaturesfit,dichotomous,μ0,dichotomous[i],τ0,param,varϵ)
        end

        # compute yfit, β at optimized τ,μ, on the full sample
        gL  = sigmoidf(x[:,i],μ,τ,param.sigmoid,dichotomous=dichotomous[i])
        G   = updateG_allocated(G0,gL,Matrix{T}(undef,n,2^depth))

        loss,yfit,β = fitβ(r,h,G,param,varϵ,infeaturesfit,dichotomous,μ,τ,dichotomous[i])

        # compute feature importance: decrease in mse
        fi2[depth] =( sum(yfit.^2) - sum(yfit0.^2) )/n

        # update matrices
        G0, loss0, yfit0, βfit = G, loss, yfit, β
        ifit, μfit, τfit, βfit  = vcat(ifit,i),vcat(μfit,μ),vcat(τfit,τ), β

    end

    return yfit0,ifit,μfit,τfit,βfit,fi2
end



function updateSMARTtrees!(SMARTtrees,Gβ,tree,rh,iter)

  T   = typeof(Gβ[1])
  n, depth = length(Gβ), length(tree.i)

  SMARTtrees.gammafit   = SMARTtrees.gammafit + SMARTtrees.param.lambda*Gβ
  SMARTtrees.infeatures = updateinfeatures(SMARTtrees.infeatures,tree.i)
  push!(SMARTtrees.trees,tree)

  for d in 1:depth
      SMARTtrees.fi2[tree.i[d]]  = SMARTtrees.fi2[tree.i[d]] + tree.fi2[d]
  end

  # R2 computed on WEIGHTED pseudo-residuals: g/sqrt(h) is N( (G*beta)*hs  ,vare )
  if iter==1 && SMARTtrees.param.R2p==T(0.898)
      sqrth    = sqrt.(rh.h)
      R2tree   = var(Gβ.*sqrth)/var((rh.r)./sqrth)
      SMARTtrees.param.R2p = R2tree
  end

end




function SMARTtreebuild(x::AbstractMatrix{T},ij,μj::AbstractVector{T},τj::AbstractVector{T},βj::AbstractVector{T},sigmoid)::AbstractVector{T} where T<:AbstractFloat

    n,p   = size(x)
    depth = length(ij)
    G     = ones(T,n)

    for d in 1:depth
        i, μ, τ = ij[d], μj[d], τj[d]
        gL  = sigmoidf(x[:,i],μ,τ,sigmoid)  # leaving dichotomous to false is a (very accurate) approximation which relies on x de-meaned and very high tau
        G   = updateG(G,gL)
    end

    gammafit = G*βj

    return gammafit
end
