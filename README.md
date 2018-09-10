# Biogeography-based optimization (BBO)
This code is an implementation of BBO algorithm to solve the famous travelling salesman(TSP) problem of 5 cities.
Biogeography-based optimization (BBO) is an evolutionary algorithm (EA) that optimizes a function by stochastically a
nd iteratively improving candidate solutions with regard to a given measure of quality, or fitness function. BBO belongs
to the class of metaheuristics since it includes many variations, and since it does not make any assumptions about the 
problem and can therefore be applied to a wide class of problems.

BBO is typically used to optimize multidimensional real-valued functions, but it does not use the gradient of the function, 
which means that it does not require the function to be differentiable as required by classic optimization methods such as 
gradient descent and quasi-newton methods. BBO can therefore be used on discontinuous functions.

BBO optimizes a problem by maintaining a population of candidate solutions, and creating new candidate solutions by combining
existing ones according to a simple formula. In this way the objective function is treated as a black box that merely provides 
a measure of quality given a candidate solution, and the function's gradient is not needed.