#
#  Main functions that are exported.
#
#  CALIBRATING PRIOR
#  SMARTloglikdivide       calibrates param.loglikdivide for panel data and/or overlapping data
#
#  FITTING and FORECASTING
#  SMARTfit                fits SMARTboost with cv (or validation/early stopping) of number of trees and, optionally, depth or other parameter
#  SMARTbst                fits SMART when nothing needs to be cross-validated (not even number of trees)
#  SMARTpredict            prediction from SMARTtrees::SMARTboostTrees
#
#  POST-ESTIMATION ANALYSIS
#  SMARTrelevance          computes feature importance (Breiman et al 1984 relevance)
#  SMARTpartialplot        partial dependence plots (keeping all other features fixed, not integrating out)
#  SMARTmarginaleffects    provisional! Numerical computation of marginal effects.
#

"""
    SMARTloglikdivide(df,y_symbol,date_symbol;overlap=0)
Suggests a value for param.loglikdivide (where param::SMARTparam). sample size/loglikedivide = effective sample size.
The only effect of loglikdivide in SMARTboost is to calibrate the strength of the prior in relation to the likelihood evidence.
The value obtained from this function can also be used as the starting value in a cross-validation search.
Accounts (roughly) for cross-sectional correlation using a clustered standard errors approach, and for serial correlation induced
by overlapping observation when y(t) = Y(t+horizon) - Y(t).

# Inputs
- `df::DataFrame`        dataframe including y and dates
- `y_symbol::Symbol`     symbol of the dependent variable in the dataframe
- `date_symbol::Symbol`  symbol of date column in the dataframe
- `overlap::Int`         (keyword) [0] horizon = number of overlaps + 1

# Output
- `loglikdivide::Float`
- `effective_sample_size::Float`

# Example of use
    lld,ess =  SMARTloglikdivide(df,:excessret,:date,overlap=h-1)
    param   =  SMARTparam(loglikdivide=lld)

"""
function SMARTloglikdivide(df::DataFrame,y_symbol::Symbol,date_symbol::Symbol; overlap::Int = 0)

    dates             = unique(df.date)
    y                 = df[:,y_symbol] .- mean(df[:,y_symbol])
    ssc       = 0.0

    for date in dates
        ssc   = ssc + (sum(y[df.date.==date]))^2
    end

    loglikdivide  = ssc/sum(y.^2)   # roughly accounts for cross-correlation as in clustered standard errors.
    
    if loglikdivide<0.9
      @warn "loglikdivide is calculated to be $loglikdivide (excluding any overlap). Numbers smaller than one imply negative cross-correlation, perhaps induced by output transformation (e.g. from y to rank(y)).
      loglikvidide WILL BE SET TO 1.0 by default. If the negative cross-correlation is genuine, the original value of $loglikdivide can be used, which would imply weaker priors."
      loglikdivide = 1.0
    end

    loglikdivide  = loglikdivide*( 1 + overlap/2 ) # roughly accounts for auto-correlation induced by overlapping, e.g. y(t) = p(t+h) - p(t)
    effective_sample_size = length(y)/loglikdivide

   return loglikdivide,effective_sample_size
end



"""
    SMARTbst(data::SMARTdata, param::SMARTparam)
SMARTboost fit, number of trees defined by param.ntrees, not cross-validated.

# Output
- `SMARTtrees::SMARTboostTrees`
"""
function SMARTbst(data::SMARTdata, param::SMARTparam )

    # initialize SMARTtrees
    data,meanx,stdx                 = preparedataSMART(data,param)
    τgrid,μgrid,dichotomous,n,p     = preparegridsSMART(data,param)
    gamma0                          = initialize_gamma(data,param)
    gammafit                        = fill(gamma0,n)

    rh                 = evaluate_pseudoresid( data.y,gammafit,param )
    SMARTtrees         = SMARTboostTrees(param,gamma0,n,p,meanx,stdx)

    for iter in 1:param.ntrees
        Gβ,i,μ,τ,β,fi2  = fit_one_tree(rh.r,rh.h,data.x,SMARTtrees.infeatures,μgrid,dichotomous,τgrid,param)

        tree              = SMARTtree(i,μ,τ,β,fi2)
        updateSMARTtrees!(SMARTtrees,Gβ,tree,rh,iter)
        rh               = evaluate_pseudoresid( data.y, SMARTtrees.gammafit, param )
    end

    return SMARTtrees
end


function displayinfo(verbose::Symbol,iter::Int,meanloss_iter,stdeloss_iter)
    if verbose == :On
        println("Tree number ", iter, "  mean and standard error of validation loss ", [meanloss_iter, stdeloss_iter])
    end
end


"""
    SMARTpredict(x,SMARTtrees::SMARTboostTrees)
Forecasts from SMARTboost

# Inputs
- `x`                           (n,p) matrix of forecast origins (type<:real)
- `SMARTtrees::SMARTboostTrees` from previously fitted SMARTbst or SMARTfit

# Output
- `yfit`                        (n) vector of forecasts of y (or, outside regression, of the natural parameter)

# Example of use
    output = SMARTfit(data,param)
    yf     = SMARTpredict(x_oos,output.SMARTtrees)
"""
function SMARTpredict(x,SMARTtrees::SMARTboostTrees)

    T       = typeof(SMARTtrees.gamma0)
    x       = convert(Matrix{T},x)

    x        = (x .- SMARTtrees.meanx)./SMARTtrees.stdx
    gammafit = SMARTtrees.gamma0*ones(T,size(x,1))

    for j in 1:length(SMARTtrees.trees)
        tree     =  SMARTtrees.trees[j]
        gammafit += SMARTtrees.param.lambda*SMARTtreebuild(x,tree.i,tree.μ,tree.τ,tree.β,SMARTtrees.param.sigmoid)
    end

    return gammafit
end



"""
    SMARTfit(data,param;paramfield=:depth,cv_grid=[],lossf=:default,nofullsample=false)

Fits SMARTboost with with n-fold cross-validation (or validation/early stopping) of number of trees and, optionally, of depth or some other field
in param. Default cross-validates only the number of trees (ntrees should not be a field.)

# Inputs
- `data::SMARTdata`
- `param::SMARTparam`
- `paramfield::Symbol`       (keyword) [:depth]  which field in param to cross-validated (besides ntrees)
- `cv_grid::AbstractVector`  (keyword) [] vector of values of paramfield
- `lossf::Symbol`            (keyword) [:default] loss function for cross-validation (:mse,:mae,:rmse); the default is mse for L2 loss
- `stopwhenlossup::Bool`     [false] in cv over paramfield, stops as soon as loss increases.
- `nofullsample::Bool`       (keyword) [false] if true AND param.nfold = 1, SMARTboost is not re-estimated on the full sample after cross-validation
                            of ntrees and (optionally) paramfield on the training sample. Reduces computing time by roughly 60% when nfold = 1,
                            at the cost of a modest loss of efficiency. Useful for very large datasets, in prelimiary analysis, and in simulations.

# Output (named tuple)

- `indtest::Vector{Vector{Int}}`  (keyword) indexes of validation samples
- `bestvalue::Float`              (keyword) best value of paramfield in cv_grid; if cv_grid==[], this is the field in param
- `ntrees::Int`                   (keyword) number of trees (best value of param.ntrees)
- `loss::Float`                   (keyword) best cv loss (best value of ntrees and paramfield)
- `meanloss:Vector{Float}`        (keyword) mean cv loss at bestvalue (of paramfield) for param.ntrees = 1,2,....
- `stdeloss:Vector{Float}`        (keyword) standard errror of cv loss at bestvalue (of paramfield) for param.ntrees = 1,2,....
- `lossgrid::Vector{Float}`       (keyword) cv loss for best tree size for each grid value
- `SMARTtrees::SMARTboostTrees`   (keyword) for the best cv value of paramfield and ntrees

# Notes
- The following options for cross-validation are specified in param: randomizecv, nfold, sharevalidation, stderulestop

# Examples of use:
    output = SMARTfit(data,param)
    output = SMARTfit(data,param;paramfield=:depth,cv_grid=[1,2,3,4],lossf=:mae)
    output = SMARTfit(data,param;paramfield=:lambda,cv_grid=[0.02,0.2])
"""
function SMARTfit( data::SMARTdata, param::SMARTparam;paramfield::Symbol = :depth, cv_grid = [],
    lossf::Symbol = :default, nofullsample::Bool = false, stopwhenlossup::Bool=false )

    T = typeof(param.varμ)
    I = typeof(param.nfold)

    if length(cv_grid)==0
        cv_grid = [param.depth]
    end

    param0                = deepcopy(param)
    fieldvalue0           = getfield(param,paramfield)
    cv_grid               = typeof(fieldvalue0).(cv_grid)  # ensure cv_grid is of the correct type

    treesize, lossgrid    = Array{I}(undef,length(cv_grid)), fill(T(Inf),length(cv_grid))
    meanloss_a,stdeloss_a = Array{Array{T}}(undef,length(cv_grid)), Array{Array{T}}(undef,length(cv_grid))
    SMARTtrees_a          = Array{SMARTboostTrees}(undef,length(cv_grid))
    indtest_a             = Vector{Vector{I}}


    for (i,fieldvalue) in enumerate(cv_grid)

        param = deepcopy(param0)
        setfield!(param, Symbol(paramfield), fieldvalue ) # Symbol() is redundant since paramfield is a symbol; needed only if paramfiels is a string
        ntrees,loss,meanloss,stdeloss,SMARTtrees1st,indtest = SMARTsequentialcv( data, param, lossf = lossf)

        treesize[i], lossgrid[i]      = ntrees, loss
        meanloss_a[i], stdeloss_a[i]  = meanloss,stdeloss
        SMARTtrees_a[i]               = SMARTtrees1st
        indtest_a                     = indtest    # same for all i

        if stopwhenlossup && i>1 && meanloss_a[i]>meanloss_a[i-1]; break; end;

    end

    best_i         = argmin(lossgrid)
    bestvalue      = cv_grid[best_i]
    ntrees         = treesize[best_i]
    loss           = lossgrid[best_i]
    SMARTtreesCV   = SMARTtrees_a[best_i]
    meanloss       = meanloss_a[best_i]
    stdeloss       = stdeloss_a[best_i]

    # Having selected best parameters by CV, fit again on the full sample (unlss param.nfold=1 and nofullsample=true)
    if nofullsample==false || param.nfold>1
        param = deepcopy(param0)
        param.ntrees   = ntrees
        if typeof(bestvalue)<:AbstractFloat; Tf = T; elseif typeof(bestvalue)<:Int; Tf = I; else; Tf = typeof(bestvalue); end;
        setfield!(param, Symbol(paramfield), Tf(bestvalue) )
        SMARTtrees       = SMARTbst(data, param )
    else
        SMARTtrees = deepcopy(SMARTtreesCV)
    end

    param = deepcopy(param0)

    return ( indtest = indtest_a, bestvalue=bestvalue,ntrees=ntrees,loss=loss,meanloss=meanloss,stdeloss=stdeloss,lossgrid=lossgrid,SMARTtrees=SMARTtrees)

end


"""
    SMARTrelevance(SMARTtrees::SMARTboostTrees,data::SMARTdata;verbose = true)

Computes feature importance (summing to 100), defined by the relevance measure of Breiman et al. (1984), equation 10.42 in
Hastie et al., "The Elements of Statistical Learning", second edition, except that the normalization is for sum = 100, not for largest = 100.
Relevance is defined on the fit of the trees on pseudo-residuals.

# Output
- `fnames::Vector{String}`         feature names, same order as in data
- `fi::Vector{Float}`              feature importance, same order as in data
- `fnames_sorted::Vector{String}`  feature names, sorted from highest to lowest importance
- `fi_sorted::Vector{Float}`       feature importance, sorted from highest to lowest
- `sortedindx::Vector{Int}`        feature indices, sorted from highest to lowest importance

# Example of use
    output = SMARTfit(data,param)
    fnames,fi,fnames_sorted,fi_sorted,sortedindx = SMARTrelevance(output.SMARTtrees,data,verbose = false)
"""
function SMARTrelevance(SMARTtrees::SMARTboostTrees,data::SMARTdata; verbose = true )

    fnames     = data.fnames
    fi         = sqrt.(abs.(SMARTtrees.fi2.*(SMARTtrees.fi2 .>=0.0) ))   # feature importance. Ocassional (tiny) negative numbers set to zero
    fi         = 100.0*fi./sum(fi)
    sortedindx = sortperm(fi,rev = true)

#    verbose == true ? printmat(hcat(fnames[sortedindx],fi[sortedindx])) :
    if verbose == true
        m   = Matrix(undef,size(data.x)[2],2)
        m[:,1] = fnames[sortedindx]
        m[:,2] = fi[sortedindx]
        printlnPs("\nFeature relevance, sorted from highest to lowest, adding up to 100 \n")
        printmat(m)                          # Paul Soderlind's printmat()
    end

    return fnames,fi,fnames[sortedindx],fi[sortedindx],sortedindx
end




"""
    SMARTpartialplot(data::SMARTdata,SMARTtrees::SMARTboostTrees,features::Vector{Int64};other_xs::Vector=[],q1st=0.01,npoints=1000))
Partial dependence plot for selected features.
For feature i, computes f(x_i) for x_i between q1st and 1-q1st quantile, with all other features at their mean (or other value x_s).

# Inputs

- `data::SMARTdata`
- `SMARTtrees::SMARTboostTrees`
- `features::Vector{Int}`        position index (in data.x) of features to compute partial dependence plot for
- `other_xs::Vector{Float}`      (keyword), a size(data.x)[1] vector of values at which to evaluate the responses. []
- `q1st::Float`                  (keyword) first quantile to compute, e.g. 0.001. Last quantile is 1-q1st. [0.01]
- `npoints::Int'                 (keyword) number of points at which to evalute f(x). [1000]

# Output
- `q::Matrix`                   (npoints,length(features)), values of x_i at which f(x_i) is evaluated
- `pdp::Matrix`                 (npoints,length(features)), values of f(x_i)

# Example
    output = SMARTfit(data,param)
    q,pdp  = SMARTpartialplot(data,output.SMARTtrees,[1,3])

# Example
    output = SMARTfit(data,param)
    fnames,fi,fnames_sorted,fi_sorted,sortedindx = SMARTrelevance(output.SMARTtrees,data,verbose=false)
    q,pdp  = SMARTpartialplot(data,output.SMARTtrees,sortedindx[1,2],q1st=0.001)
"""
function SMARTpartialplot(data::SMARTdata,SMARTtrees::SMARTboostTrees,features::Vector{Int64};other_xs::Vector =[],q1st=0.01,npoints = 1000)

    T = typeof(SMARTtrees.param.lambda)

    if length(other_xs)==0
        other_xs =  T.(mean(data.x,dims=1))
    else
        other_xs  = T.(other_xs')
    end

    step = (1-2*q1st)/(npoints-1)
    p    = [i for i in q1st:step:1-q1st]

    pdp = Matrix{T}(undef,length(p),length(features))
    q  = Matrix{T}(undef,length(p),length(features))

    for (i,f) in enumerate(features)
        q[:,i] = T.(quantile(data.x[:,f],p))
        h = ones(T,length(p)).*other_xs
        h[:,f] = q[:,i]
        pdp[:,i] = SMARTpredict(h,SMARTtrees)
    end

    return q,pdp
end


"""
    SMARTmarginaleffect(data::SMARTdata,SMARTtrees::SMARTboostTrees,features::Vector{Int64};other_xs::Vector =[],q1st=0.01,npoints=50)
APPROXIMATE Computation of marginal effects using NUMERICAL derivatives.

# Inputs

- `data::SMARTdata`
- `SMARTtrees::SMARTboostTrees`
- `features::Vector{Int}`        position index (in data.x) of features to compute partial dependence plot for
- `other_xs::Vector{Float}`      (keyword), a size(data.x)[1] vector of values at which to evaluate the marginal effect. []
- `q1st::Float`                  (keyword) first quantile to compute, e.g. 0.001. Last quantile is 1-q1st. [0.01]
- `npoints::Int'                 (keyword) number of points at which to evalute df(x_i)/dx_i. [50]

# Output
- `q::Matrix`                   (npoints,length(features)), values of x_i at which f(x_i) is evaluated
- `d::Matrix`                   (npoints,length(features)), values of marginal effects

# NOTE: Provisional! APPROXIMATE Computation of marginal effects using NUMERICAL derivatives.

# Example
    output = SMARTfit(data,param)
    q,d    = SMARTmarginaleffect(data,output.SMARTtrees,[1,3])

# Example
    output = SMARTfit(data,param)
    fnames,fi,fnames_sorted,fi_sorted,sortedindx = SMARTrelevance(output.SMARTtrees,data,verbose=false)
    q,pdp  = SMARTmarginaleffect(data,output.SMARTtrees,sortedindx[1,2],q1st=0.001)

"""
function SMARTmarginaleffect(data::SMARTdata,SMARTtrees::SMARTboostTrees,features::Vector{Int64};other_xs::Vector =[],q1st=0.01,npoints = 50)

    # compute a numerical derivative
    q,pdp   = SMARTpartialplot(data,SMARTtrees,features,other_xs = other_xs,q1st = q1st,npoints = npoints+2)
    n       = size(q,1)
    d       = (pdp[1:n-2,:] - pdp[3:n,:])./(q[1:n-2,:] - q[3:n,:] )  # numerical derivatives at q[i]: f(q[i+1]-f(q[i-1])/(q[i+1]-q[i-1]) )

    return q[2:n-1,:],d
end
