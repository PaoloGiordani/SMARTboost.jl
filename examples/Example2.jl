"""
Example2

Global equity data (monthly total excess returns for equity indexes )

- Prepare panel data, originally in DataFrames format, for SMARTboost:
  1) sort dataframe by date; necessary for a panel, typically advsisable for a single time series as well
  2) convert dataframe columns into a vector of floats y and a matrix of floats x
  3) calibrate loglikdivide, lld = SMARTloglikdivide(...,overlap = ...)
- param = SMARTparam(loglikdivide = lld,overlap = ... )
- fit, optionally cross-validating depth
- save fitted model (upload fitted model)
- feature importance
- partial effects plots
- marginal effects

paolo.giordani@bi.no
"""

# On one core
# using SMARTboost

# On multiple cores
number_workers  = 4  # desired number of workers
using Distributed
nprocs()<number_workers ? addprocs( number_workers - nprocs()  ) : addprocs(0)
@everywhere using SMARTboost

using Random, Plots, CSV, DataFrames, Statistics

Random.seed!(12345)

log_ret        = true    # true for log returns, fase for returns

# some options for SMARTboost
cvdepth        = false      # false to default to depth = 3
cvgrid         = [1,2,3,4]

# END USER'S OPTIONS
df = CSV.read("examples/data/GlobalEquityReturns.csv", DataFrame, copycols = true) # import data as dataframe. Monthly LOG excess returns.
display(describe(df))

# prepare data for SMART. Sort dataframe by date
sort!(df,:date)

features_vector = [:logCAPE, :momentum, :vol3m, :vol12m ]
log_ret ? y     = 100*df[:,:excessret] : y  = @. 100*(exp(df[:,:excessret]) - 1.0 )

lld,ess =  SMARTloglikdivide(df,:excessret,:date,overlap=0)
println("\n loglikdivided calibrated as ", lld)
println(" nominal and effective samples size  ", [length(y),ess])

# set up SMARTparam and SMARTdata, then fit
fnames = ["logCAPE", "momentum", "vol3m", "vol12m"  ]
param  = SMARTparam(loglikdivide = lld,overlap=0,stopwhenlossup=true) # in CV, stop as soon as loss increases
data   = SMARTdata(y,df[:,features_vector],param,df[:,:date],fnames = fnames)

if cvdepth   # stop as soon as loss increases
    output = SMARTfit(data,param,paramfield=:depth,cv_grid = cv_grid)
else
    output = SMARTfit(data,param)
end

# Two ways of extracting in-sample fitted value. yfit = yfit2 (up to rounding errors)
yfit      = SMARTpredict(data.x,output.SMARTtrees)
yfit2     = output.SMARTtrees.gammafit

# Compare with ols
xols        = hcat( ones(length(data.y)),data.x )
β           = (xols'xols)\(xols'data.y)
yfit_ols    = xols*β

println("\n depth = $(output.bestvalue), number of trees = $(output.ntrees) ")
println(" correlation of SMART and ols fitted values (in-sample) ", cor(yfit,yfit_ols) )


# save (load) fitted model
#@save "output.jld2" output
#@load "output.jld2" output    # Note: key must be the same, e.g. @load "output.jld2" output2 is a KeyError

# feature importance
fnames,fi,fnames_sorted,fi_sorted,sortedindx = SMARTrelevance(output.SMARTtrees,data)

# partial dependence plots, best four features. q1st is the first quantile. e.g. 0.01 or 0.05
q,pdp  = SMARTpartialplot(data,output.SMARTtrees,sortedindx[[1,2,3,4]],q1st=0.01,npoints = 5000)

p1   = plot(q[:,1],pdp[:,1], label = "$(fnames[sortedindx[1]])",legend=:topright,color=:green)
p2   = plot(q[:,2],pdp[:,2], label = "$(fnames[sortedindx[2]])",legend=:topright,color=:green)
p3   = plot(q[:,3],pdp[:,3], label = "$(fnames[sortedindx[3]])",legend=:topright,color=:green)
p4   = plot(q[:,4],pdp[:,4], label = "$(fnames[sortedindx[4]])",legend=:topright,color=:green)
display(plot(p1,p2,p3,p4, layout=(2,2), size=(1200,600)))  # display() will show it in Plots window.

# partial response of all features
pl = Vector(undef,4)

j  = 2    # which feature has high-low-medium values. 2 for momentum
for i in 1:4

  other_xs    = vec(mean(data.x,dims=1))
  q,pdp_med  = SMARTpartialplot(data,output.SMARTtrees,[i],q1st=0.01,npoints = 5000,other_xs=other_xs)
  other_xs[j] = quantile(x[:,j],0.9) # other_xs[j] = maximum(x[:,j])
  q,pdp_high  = SMARTpartialplot(data,output.SMARTtrees,[i],q1st=0.01,npoints = 5000,other_xs=other_xs)
  other_xs[j] = quantile(x[:,j],0.1) # other_xs[j] = minimum(x[:,j])
  q,pdp_low  = SMARTpartialplot(data,output.SMARTtrees,[i],q1st=0.01,npoints = 5000,other_xs=other_xs)

  pdp  = hcat(pdp_med,pdp_high,pdp_low)
  pl[i]   = plot( q[:,1],pdp,title = "$(fnames[i]) for high, avg, low $(fnames[j])",
       label = ["medium" "high" "low"],
       legend = :bottomleft,
       linecolor = [:blue :green :red],
       linestyle = [:solid :dot :dot],

       linewidth = [5 3 3],
       titlefont = font(15),
       legendfont = font(12),
       xlabel = "$(fnames[i])",
       ylabel = "expected returns",
       )

end

display(plot(pl[1], pl[2], pl[3], pl[4], layout=(2,2), size=(1200,800)))  # display() will show it in Plots window.


# marginal effects (note: numerical derivatives, can be improved by taking analytical derivatives)
pl = Vector(undef,4)

for i in 1:4

  other_xs    = vec(mean(data.x,dims=1))
  npoints     = 200
  q,me_med  = SMARTmarginaleffect(data,output.SMARTtrees,[i],q1st=0.01,npoints = npoints,other_xs=other_xs)
  other_xs[j] = quantile(x[:,j],0.9) # other_xs[j] = maximum(x[:,j])
  q,me_high  = SMARTmarginaleffect(data,output.SMARTtrees,[i],q1st=0.01,npoints = npoints,other_xs=other_xs)
  other_xs[j] = quantile(x[:,j],0.1) # other_xs[j] = minimum(x[:,j])
  q,me_low  = SMARTmarginaleffect(data,output.SMARTtrees,[i],q1st=0.01,npoints = npoints,other_xs=other_xs)

  me  = hcat(me_med,me_high,me_low)
  pl[i]   = plot( q[:,1],me,title = "$(fnames[i]) for high, avg, low $(fnames[j])",
       label = ["medium" "high" "low"],
       legend = :bottomleft,
       linecolor = [:blue :green :red],
       linestyle = [:solid :dot :dot],

       linewidth = [5 3 3],
      # title = "linear dgp",
       titlefont = font(15),
       legendfont = font(12),
       xlabel = "$(fnames[i])",
       ylabel = "marginal effect",
       )

end

display(plot(pl[1], pl[2], pl[3], pl[4], layout=(2,2), size=(1200,800)))  # display() will show it in Plots window.
