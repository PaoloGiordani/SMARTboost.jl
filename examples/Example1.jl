
"""
Example 1

Simulated iid data, additively nonlinear dgp.

- :L2 or :logistic. If :logistic, the fitted and forecast values are for the log odds ratio
- fit, optionally cross-validating depth
- save fitted model (upload fitted model)
- feature importance
- partial effects plots
- marginal effects

paolo.giordani@bi.no
"""

# On one core
using SMARTboost

# On multiple cores
#=
number_workers  = 8  # desired number of workers
using Distributed
nprocs()<number_workers ? addprocs( number_workers - nprocs()  ) : addprocs(0)
@everywhere using SMARTboost
=#

using Random, JLD2, Plots

Random.seed!(12345)

# Some options for SMARTboost
cvdepth   = false    # false to use the default depth (3), true to cv
nfold     = 1        # nfold cv. 1 faster (singl validation sets), default 5 is slower, but more accurate.

# options to generate data. y = sum of four additive nonlinear functions + Gaussian noise(0,stde^2)
n,p,n_test  = 1_000,10,100_000
stde        = 1.0

f_1(x,b)    = b*x
f_2(x,b)    = sin.(b*x)  # for higher nonlinearities, try #f_2(x,b) = 2*sin.(2.5*b*x)
f_3(x,b)    = b*x.^3
f_4(x,b)    = b./(1.0 .+ (exp.(4.0*x))) .- 0.5*b

b1,b2,b3,b4 = 1.5,2.0,0.5,2.0

# END USER'S OPTIONS
# generate data
x,x_test = randn(n,p), randn(n_test,p)
f        = f_1(x[:,1],b1) + f_2(x[:,2],b2) + f_3(x[:,3],b3) + f_4(x[:,4],b4)
f_test   = f_1(x_test[:,1],b1) + f_2(x_test[:,2],b2) + f_3(x_test[:,3],b3) + f_4(x_test[:,4],b4)
y        = f + stde*randn(n)

# set up SMARTparam and SMARTdata, then fit and predit
param  = SMARTparam(nfold = nfold)
data   = SMARTdata(y,x,param,fdgp=f)

if cvdepth==false
    output = SMARTfit(data,param)                # default depth
else
    output = SMARTfit(data,param,paramfield=:depth,cv_grid=[1,2,3,4])
end

yf     = SMARTpredict(x_test,output.SMARTtrees)  # predict

println("\n depth = $(output.bestvalue), number of trees = $(output.ntrees) ")
println(" out-of-sample RMSE from truth ", sqrt(sum((yf - f_test).^2)/n_test) )

# save (load) fitted model
#@save "output.jld2" output
#@load "output.jld2" output    # Note: key must be the same, e.g. @load "output.jld2" output2 is a KeyError

# feature importance, partial dependence plots and marginal effects
#fnames,fi,fnames_sorted,fi_sorted,sortedindx = SMARTrelevance(output.SMARTtrees,data)
q,pdp  = SMARTpartialplot(data,output.SMARTtrees,[1,2,3,4])
qm,me  = SMARTmarginaleffect(data,output.SMARTtrees,[1,2,3,4],npoints = 40)

display(plot(output.meanloss[1:end],title = "cv loss",label = "cv loss"))

# plot partial dependence
pl   = Vector(undef,4)
f,b  = [f_1,f_2,f_3,f_4],[b1,b2,b3,b4]

for i in 1:length(pl)
    pl[i]   = plot( [q[:,i]],[pdp[:,i] f[i](q[:,i],b[i])],
           label = ["smart" "dgp"],
           legend = :bottomright,
           linecolor = [:blue :red],
           linestyle = [:solid :dot],

           linewidth = [5 5],
           titlefont = font(15),
           legendfont = font(12),
           xlabel = "x",
           ylabel = "f(x)",
           )
end

display(plot(pl[1], pl[2], pl[3], pl[4], layout=(2,2), size=(1300,800)))  # display() will show it in Plots window.
