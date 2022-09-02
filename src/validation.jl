#
# Functions for validation (early stopping) and CV
#
# indices_from_dates     indices for cumulative (expanding window CV).
# indicescv_panel_purge  indices for purged cv. Extends De Prado to panel data.
# add_tree
# SMARTsequentialcv     key function: fits nfold sequences of boosted trees, and selects number of trees by cv. Computationally efficient validation/cv of number of trees in boosting

"""
    indexes_from_dates(df::DataFrame,datesymbol::Symbol,first_date::Date,n_reestimate::Int)

Computes indexes of training set and test set for cumulative CV and pseudo-real-time forecasting exercises

* INPUTS

- datesymbol            symbol name of the date
- first_date            when the first training set ENDS (end date of the first training set)
- n_reestimate          every how many periods to re-estimate (update the training set)

* OUTPUT
  indtrain_a,indtest_a are arrays of arrays of indexes of train and test samples

* Example of use

- first_date = Date("2017-12-31", Dates.DateFormat("y-m-d"))
- indtrain_a,indtest_a = indexes_from_dates(df,:date,first_date,12)

* NOTES

- Inefficient for large datasets

"""
function indexes_from_dates(df::DataFrame,datesymbol::Symbol,first_date::Date,n_reestimate::Int)

    indtrain_a = Vector{Int}[]
    indtest_a  = Vector{Int}[]

    dates   = df[:,datesymbol]
    datesu  = unique(dates)
    date1   = first_date    # end date of training set

    N       = length(datesu)
    i       = length(datesu[datesu.<=first_date])
    date2   = datesu[i+n_reestimate]   # end date of test set
    finish  = false

    while finish == false

        df_train   = df[df[:,datesymbol].<= date1,:]
        indtrain   = Vector{Int}[collect(1:size(df_train,1))]
        df_test    = df[(df[:,datesymbol].> date1).*(df[:,datesymbol].<= date2),:]
        indtest    = Vector{Int}[indtrain[end][end] .+ collect(1:size(df_test,1))]

        i      = i+n_reestimate
        if i >= N
            finish = true
        else
            date1  = datesu[i]
            date2  = datesu[minimum(hcat(N,i+n_reestimate))]
            indtrain_a = append!(indtrain_a,indtrain)
            indtest_a  = append!(indtest_a,indtest)
        end

    end

    return indtrain_a,indtest_a
end

"""
    indicescv(j,nfold,sharevalidation,n,indices)
indtrain,indtest = indicescv(j,nfold,sharevalidation,n,indices)
Intended use: ytrain = data.y[indtrain], xtrain = data.x[indtrain,:], ytest = data.y[indtest], xtest = data.x[indtest,:]
indices = [i for i in 1:n] for contiguous blocks, indices = shuffle([i for i = 1:n]) for randomized train and test data
"""
function indicescv(j::I,nfold::I,sharevalidation,n::I,indices::Vector{I}) where I<:Int  # j = 1,...,nfold

    if nfold == I(1)
        typeof(sharevalidation)==I ? samplesize = sharevalidation : samplesize = I(floor(n*sharevalidation))
        starti     = n - samplesize + 1
        indtest    = indices[[i for i in starti:n]]
        indtrain   = indices[[i for i in 1:starti-1]]
    else
        samplesize = I(floor(n/nfold))
        starti     = I(1+(samplesize*(j-1)))
        j==nfold ? endi = n : endi   = samplesize*j
        indtest       = indices[[i for i in starti:endi]]

        if j==1
            indtrain = indices[[i for i in endi+1:n]]
        elseif j==nfold
            indtrain = indices[[i for i in 1:starti-1]]
        else
            indtrain = vcat(indices[[i for i in 1:starti-1]],indices[[i for i in endi+1:n]])
        end

    end

    return indtrain,indtest

end


"""
    indicescv_panel_purge(j,nfold,sharevalidation,n,indices,dates,param.overlap)

indtrain,indtest = indicescv_panel_purge(j,nfold,sharevalidation,n,indices,dates,param.overlap)

Extends to panel data the "purged cross-validation" of De Prado (which in turns is closely related to hv-Blocked CV of Racine) -- proposed for a single time series ---
to allow for both overlapping and/or panel data features (several observations sharing the same date.)
For a good academic reference, see Oliveira et al. (2021), "Evaluation Procedures for Forecasting with Spatiotemporal Data"

Sets the test sample with no concern for the date, and then purges from the training set observations with the same date. If overlap > 0, purges additional dates.
Not the most statistically efficient way to split the data, which would be to have test set start at the first observation of a new date (and end at the last), but
the loss of efficiency should be small. All data points are part of a test set (most important), but the training sets are not as large as they could be.

"""
function indicescv_panel_purge(j::I,nfold::I,sharevalidation,n::I,indices::Vector{I},dates::AbstractVector,overlap::I)  where I<:Int  # j = 1,...,nfold)

    indtrain,indtest  = indicescv(j,nfold,sharevalidation,n,indices)  # plain-vanilla split

    dateI   = dates[indtest[1]]
    dateF   = dates[indtest[end]]
    datesu  = unique(dates)

    if length(datesu)==length(dates) && overlap==I(0)
        return indtrain,indtest
    else                                       # purge from train set all indexes sharing the same date with test set, which is assumed ordered
        indtrain = hcat(indtrain,dates[indtrain])
        indtrain = indtrain[indtrain[:,2] .!= dateI,:]
        indtrain = indtrain[indtrain[:,2] .!= dateF,:]

        if overlap>I(0)                   # If there is overlap, purge more dates
            tI     = argmax(datesu .== dateI)  # index of dateI at datesu. The expression in parenthesis is only true for one value
            tF     = argmax(datesu .== dateF)  # index of dateF at datesu

            for o = 1:overlap
                if j>1
                    dateI   = datesu[tI-o]
                    indtrain = indtrain[indtrain[:,2] .!= dateI,:]
                end

                if j<nfold
                    dateF    = datesu[tF+o]
                    indtrain = indtrain[indtrain[:,2] .!= dateF,:]
                end
            end
        end
        return (indtrain[:,1],indtest)
    end

end


function add_tree(iter,SMARTtrees::SMARTboostTrees,rh,x,μgrid,dichotomous,τgrid)

    Gβ,i,μ,τ,β,fi2   = fit_one_tree(rh.r,rh.h,x,SMARTtrees.infeatures,μgrid,dichotomous,τgrid,SMARTtrees.param)
    tree             = SMARTtree(i,μ,τ,β,fi2)
    updateSMARTtrees!(SMARTtrees,Gβ,tree,rh,iter)

    return SMARTtrees,i,μ,τ,β
end



"""
    SMARTsequentialcv( data::SMARTdata, param::SMARTparam; .... )

ntrees,loss,meanloss,stdeloss,SMARTtrees = SMARTsequentialcv( data::SMARTdata, param::SMARTparam; .... )

sequential cross-validation for SMART. validation (early stopping) or n-fold cv for growing ensemble, automatically selecting the cv-optimal number of trees for a given parameter vector.

# Inputs (optional)

- lossf = :default        :default to use the loss function implied by the model (used to compute gradient), else :mse, :rmse, :mae, ....
- The following options are specified in param::SMARTparam
     randomizecv,nfold, sharevalidation,stderulestop

# Output:

- ntrees                 Number of trees chosen by cv (or validation)
- loss                   Average loss in nfold test samples evaluated at ntrees
- meanloss               Vector of loss (mean across nfolds) at 1,2,...,J, where J>=ntrees
- stdeloss               Vector of stde(loss) at 1,2,...,J, where J>=ntrees; standard error of the estimated mean loss computed by aggregating loss[i] across i = 1,....,n, so std ( l .- mean(l)  )/sqrt(n).
                         Unlike the more common definition, this applies to p = 1 as well.
- SMARTtrees1st          ::SMARTboostTrees fitted on y_train,x_train for first fold (intended for nfold = 1)
- indtest                nfold vector of vectors of indices of validation sample
"""
function SMARTsequentialcv( data::SMARTdata, param::SMARTparam; lossf::Symbol = :default)

    T = typeof(param.varμ)
    I = typeof(param.depth)

    nfold, sharevalidation,stderulestop, n = param.nfold, param.sharevalidation, param.stderulestop, I(length(data.y))

    rh_a             = Array{NamedTuple}(undef,nfold)
    gammafit_test_a  = Array{Vector{T}}(undef,nfold)
    t_a              = Array{Tuple}(undef,nfold)
    SMARTtrees_a     = Array{SMARTboostTrees}(undef,nfold)
    indtest_a        = Vector{Vector{I}}(undef,nfold)

    param.randomizecv==true ? indices = shuffle([i for i = 1:n]) : indices = [i for i in 1:n]

    for nf in 1:nfold

        indtrain,indtest                      = indicescv_panel_purge(nf,nfold,sharevalidation,n,indices,data.dates,param.overlap)
        data_nf                               = SMARTdata(data.y[indtrain],data.x[indtrain,:],param,data.dates[indtrain],enforce_dates = false)
        data_nf,meanx,stdx                    = preparedataSMART(data_nf,param)

        τgrid,μgrid,dichotomous,n_train,p     = preparegridsSMART(data_nf,param)
        gamma0                                = initialize_gamma(data_nf,param)

        SMARTtrees_a[nf]    = SMARTboostTrees(param,gamma0,n_train,p,meanx,stdx)
        rh_a[nf]            = evaluate_pseudoresid(data_nf.y, SMARTtrees_a[nf].gammafit, param )
        gammafit_test_a[nf] = gamma0*ones(I,length(indtest))

        t_a[nf]             = (indtrain,indtest,meanx,stdx,n_train,p,τgrid,μgrid,dichotomous)
        indtest_a[nf]       = indtest

    end

    lossM, meanloss, stdeloss, j =  zeros(T,param.ntrees,nfold), zeros(T,param.ntrees), zeros(T,param.ntrees), I(0)

    for i in 1:param.ntrees

        lossv  = T[]

        for nf in 1:nfold

            indtrain,indtest,meanx,stdx,n_train,p,τgrid,μgrid,dichotomous  = t_a[nf]

            x_train = SharedMatrix((data.x[indtrain,:] .- meanx)./stdx)  # wasteful to do this over i .... avoids storing additional large matrices though
            x_test  = (data.x[indtest,:] .- meanx)./stdx

            SMARTtrees_a[nf],ij,μj,τj,βj = add_tree(i,SMARTtrees_a[nf],rh_a[nf],x_train,μgrid,dichotomous,τgrid)
            rh_a[nf] = evaluate_pseudoresid( data.y[indtrain], SMARTtrees_a[nf].gammafit, param )
            gammafit_test_a[nf] = gammafit_test_a[nf] + param.lambda*SMARTtreebuild(x_test,ij,μj,τj,βj,param.sigmoid)
            lossM[i,nf],losses  = lossfunction(param,lossf,data.y[indtest], gammafit_test_a[nf] )  # reverse standardization. lossv is a (ntest) vector of losses
            lossv = vcat(lossv,losses)

        end

        meanloss[i]  = mean(lossM[i,:])
        stdeloss[i]  = std( lossv .- meanloss[i],corrected = false )/sqrt(length(lossv))  # std from all observations

        displayinfo(param.verbose,i,meanloss[i],stdeloss[i])

        # break the loop if CV loss is increasing or decreasing too slowly
        if i>=20
            sdiff = (mean(meanloss[i-9:i]) - mean( meanloss[i-19:i-10])) / stdeloss[i]
        else
            sdiff = T(-Inf)
        end

        if sdiff > -stderulestop
            break
        else
            j = j+1
        end

    end

    ntrees         = argmin(meanloss[1:j])
    loss           = meanloss[ntrees]

    return ntrees,loss,meanloss[1:j],stdeloss[1:j],SMARTtrees_a[1],indtest_a

end
