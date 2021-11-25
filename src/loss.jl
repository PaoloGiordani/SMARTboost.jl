# Note: gammafit is used instead of yfit because for general loss functions/likelihoods, gamma is the natural parameter on
# gamma(i) = sum of trees. e.g. log-odds for logit.

# lossfunction is used in cv only, so it does not need to be a proper log-likelihood.
# if lossf == :default, then same loss as in evaluate_pseudoresid is used
# Other options for lossf are :mse :mae
function lossfunction(param::SMARTparam,lossf::Symbol,y,gammafit)

    T = typeof(y[1])
    n = length(y)

    if lossf == :default
        loss  = (y - gammafit).^2
    else
        if lossf == :mse
            loss  = (y - gammafit).^2
            meanloss = sum(loss)/length(y)
        elseif lossf == :rmse
            resid = y - gammafit
            loss  = abs.(resid)
            meanloss = sqrt(resid'resid/length(y))
        elseif lossf == :mae
            loss  = abs.(y - gammafit)
            meanloss = mean(loss)
        end

    end

    meanloss = sum(loss)/length(y)

    return meanloss, loss
end


function initialize_gamma(data::SMARTdata,param::SMARTparam)
  T = typeof(data.y[1])
  return T(sum(data.y)/length(data.y))
end

function evaluate_pseudoresid(y::AbstractVector{T},gammafit::AbstractVector{T},param::SMARTparam ) where T<:AbstractFloat
    r  = @. y - gammafit
    return (r = r,h = ones(T,1))
end


# defines any coefficient needed for a given loss function.
# NOTE: can only compute pure numbers; any further function of y can be called in evaluate_pseudoresid
function coeff_loss(loss::Symbol)
    coeff = [NaN]
end
