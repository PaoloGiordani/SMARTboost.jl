
"""
Speed test

Running times with different settings of n,p,cores. 100 trees.

paolo.giordani@bi.no
"""

# On one core
using SMARTboost

# On multiple cores
number_workers  = 8  # desired number of workers
using Distributed
nprocs()<number_workers ? addprocs( number_workers - nprocs()  ) : addprocs(0)
@everywhere using SMARTboost

using Random

ntrees = 100
n_a   = [1_000,100_000,1_000_000]  # @time for the very first run (here 1000) includes compile time
p_a   = [10,100]

function Simulation(n,p,ntrees)

    Random.seed!(12345)
    x    = randn(n,p)
    y     = 2*x[:,1]+x[:,2]+x[:,3]+0.5*x[:,4]+0.5*x[:,5]+randn(n)

    param  = SMARTparam(nfold=1,depth=4,ntrees=ntrees)
    data   = SMARTdata(y,x,param)
    trees = SMARTboost.SMARTbst(data,param)  # SMARTbst will run exactly ntrees

end

for p in p_a
    for n in n_a
       println("Seconds for n $n and p $p with $number_workers cores.")
       @time Simulation(n,p,ntrees)
   end
end
