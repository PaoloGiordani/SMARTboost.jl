
# options to generate data. y = sum of four additive nonlinear functions + Gaussian noise(0,stde^2)
n,p,n_test  = 1_000,5,100_000
stde        = 1.0

f_1(x,b)    = b*x
f_2(x,b)    = sin.(b*x)
f_3(x,b)    = b*x.^3
f_4(x,b)    = b./(1.0 .+ (exp.(4.0*x))) .- 0.5*b

b1,b2,b3,b4 = 1.5,2.0,0.5,2.0

# generate data
x,x_test = randn(n,p), randn(n_test,p)
f        = f_1(x[:,1],b1) + f_2(x[:,2],b2) + f_3(x[:,3],b3) + f_4(x[:,4],b4)
f_test   = f_1(x_test[:,1],b1) + f_2(x_test[:,2],b2) + f_3(x_test[:,3],b3) + f_4(x_test[:,4],b4)
y        = f + randn(n)*stde

# set up SMARTparam and SMARTdata, then fit and predit
param  = SMARTparam(nfold = 1,verbose = :Off )
data   = SMARTdata(y,x,param,fdgp=f)

output = SMARTfit(data,param,paramfield=:depth,cv_grid=[1,2,3,4]) 
yf     = SMARTpredict(x_test,output.SMARTtrees)  # predict

# feature importance, partial dependence plots and marginal effects
fnames,fi,fnames_sorted,fi_sorted,sortedindx = SMARTrelevance(output.SMARTtrees,data)
q,pdp  = SMARTpartialplot(data,output.SMARTtrees,[1,2,3,4])
qm,me  = SMARTmarginaleffect(data,output.SMARTtrees,[1,2,3,4],npoints = 40)
