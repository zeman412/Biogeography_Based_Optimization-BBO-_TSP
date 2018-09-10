
import numpy as np
import matplotlib.pyplot as plt
import random
import time
import numbers

##----------------------------------------------------------####
PopulationSize = 5
prob_mutation = 0.04
ProblemDimension = 5
max_gen = 20
mu = np.zeros(PopulationSize)
lambda1 = np.zeros(PopulationSize)

E = 1
I = 1
fit_val = np.zeros((PopulationSize),dtype=int)
Mean = np.zeros((max_gen),dtype=float)
best_fit = np.zeros((max_gen),dtype=int)

Population = np.zeros((PopulationSize, ProblemDimension), dtype=int)

def init_population(PopulationSize, ProblemDimension):

    for s in range(PopulationSize):
        Population[s] = np.random.permutation(ProblemDimension)

    return Population


distance_list =  [[0, 8, 4, 9, 9],
                 [8, 0, 6, 7, 10],
                  [4, 6, 0, 5, 6],
                  [9, 7, 5, 0, 4],
                  [9, 10, 6, 4, 0]]

##==================================================================
def fitness(solu, ProblemDimension):
    distance = 0
    for d in range(ProblemDimension):
        if (d == ProblemDimension-1):
            distance += (distance_list[solu[d]] [solu[0]])
            break
        distance += (distance_list[solu[d]] [solu[d+1]])

    return distance
# # print population before sorted
# Population = init_population(PopulationSize, ProblemDimension)
# for p in range(PopulationSize):
#     print(" Unsorted Population ", Population[p,:], "distance ", fitness(Population[p,:], ProblemDimension))

# Sort population based on fitness
def sort (Pop):
    Pop
    for i in range(PopulationSize):
     fit_val[i] = fitness(Pop[i,:], ProblemDimension)  # evaluate the cost of the candidate neighbors
    rank = np.argsort(fit_val)  # sorted index based on  cost
    Pop = Pop[rank]
    return Pop

# print sorted population
def display_pop (Population):
    for p in range(PopulationSize):
        print("Sorted Population ", Population[p,:], "distance ", fitness(Population[p,:], ProblemDimension))

def lamda_Mu(PopulationSize):
    # calculating mu and lambda
    for i in range(PopulationSize):
        mu[i] = (PopulationSize + 1 - (i)) / (PopulationSize + 1) # emigration rate
        lambda1[i] = 1 - mu[i]   # immigration rate
    return lambda1, mu
    print("Lambda value", lambda1)
    print("Mu     value", mu)

def main_BBO ():
    global Population, lambda1, mu, Island1, Island2, max_gen, best_fit, Mean
    Population = init_population(PopulationSize, ProblemDimension)
    Population = sort(Population)
    lambda1, mu = lamda_Mu(PopulationSize)
    #display_pop(Population)
    #print(Population)
    for g in range (max_gen):

        # Performing Migration operator
        delta = np.random.randint(ProblemDimension)
        for k in range(PopulationSize):
            for j in range(ProblemDimension):
                # select Hi based on mu
                if  np.random.uniform(0,1) < lambda1[k]:
                    # Performing Roulette Wheel
                    RandomNum =  np.random.uniform(0,1) * sum(mu)
                    Select = mu[0]
                    SelectIndex = 0
                    while (RandomNum > Select) and (SelectIndex < (PopulationSize - 1)):
                        SelectIndex = SelectIndex + 1
                        Select = Select + mu[SelectIndex]



                    Island1 = Population[SelectIndex, :].tolist()
                    Island2 = Population[k, :].tolist()
                    index1 = Island1.index(delta)
                    #print("Delta", delta)
                    if Island2[index1] != delta:
                        index2 = Island2.index(delta)
                        Island2[index2] = Island2[index1]
                        Island2[index1] = delta

                        Population[PopulationSize-1, :] =  Island2

        #print("Island1", Island1)
        #print("Island2", Island2)

        #print(Population)

        ### Do mutation - 2-opts - Swap Operator. note that mutation is within the same solution
        for m in range(PopulationSize):
            if np.random.uniform(0, 1) <= prob_mutation:
                RN1 = np.random.randint(ProblemDimension)
                RN2 = np.random.randint(ProblemDimension)
                while RN1 == RN2:
                    RN2 = np.random.randint(ProblemDimension)  # this could as well be RN1; just so the two are not the same

                RN1_Dep = Population[m][RN1]
                RN2_Dep = Population[m][RN2]
                Population[m][RN1] = RN2_Dep
                Population[m][RN2] = RN1_Dep
        Population = sort(Population)
        lambda1, mu = lamda_Mu(PopulationSize)
        best_fit[g] = fitness(Population[0, :], ProblemDimension)
        print("best", best_fit[g])
        print("generation", g)

        for n in range(PopulationSize):
            fit_val[n] = fitness(Population[n, :], ProblemDimension)
        print("Sorted fit", fit_val)
        for n in range(PopulationSize):
            print(Population[n, :], fitness(Population[n, :], ProblemDimension))
        Mean[g] = np.mean(fit_val)
        print("mean", Mean)

        #print(fit_val[0])
        #display_pop(Population)
       # max_gen -= 1
if __name__== "__main__":
    main_BBO ()

plt.plot(Mean)
plt.show()
# print(best_fit)
