# EDApy

## Description

This is a package in which some Estimation of Distribution Algorithms (EDAs) are implemented. EDAs are a type of evolutionary algorithms. Depending on the type of EDA, dependencies among the variables can be considered.

At the moment, three EDAs are implemented:
* Binary univariate EDA. Can be used as a simple example of EDA, or to use as a feature selection.
* Continuous univariate EDA. 
* Continuous multivariate EDA. 

## Examples

#### Binary univariate EDA
Can be used as a simple EDA or for feature selection. The cost function to optimize is the metric of the model to optimize. An example is shown.
```python
from EDApy import EDA_discrete as EDAd
import pandas as pd

def check_solution_in_model(dictionary):
    MAE = prediction_model(dictionary)
    return MAE

vector = pd.DataFrame(columns=['param1', 'param2', 'param3'])
vector.loc[0] = 0.5

EDA = EDAd(MAX_IT=200, DEAD_ITER=20, SIZE_GEN=30, ALPHA=0.7, vector=vector, 
            cost_function=check_solution_in_model, aim='minimize')

bestcost, solution, history = EDA.run(output=True)
print(bestcost, solution)
print(history)
```

The example is an implementation for feature selection (FS) for a prediction model (prediction_model). Being prediction_model a model which predicts depending on some variables. The prediction model, receives a dictionary with keys 'param_N' (N from 1 to number of parameters) and values 1 or 0 depending if variables should be included or not. The model returns a MAE which we intend to minimize.

The EDA receives as input the maximum number of iterations, the number of iterations with no best global cost improvement, the generations size, the percentage of generations to select as best individuals, the initial vector of probabilities for each variables to being chosen, the cost functions to optimize, and the aim ('minimize' or 'maximize') of the optimizer.

Vector probabilities is usually initialized to 0.5 to start from an equilibrium situation. EDA returns the best cost found, the best combination of variables, and the history os costs found, if wanted to be plotted.

#### Continuous univariate EDA

This EDA is used when some continuous parameters must be optimized. 
```python
from EDApy import EDA_continuous as EDAc
import pandas as pd
import numpy as np

wheights = [20,10,-4]

def cost_function(dictionary):
    function = wheights[0]*dictionary['param1']**2 + wheights[1]*(np.pi/dictionary['param2']) - 2 - wheights[2]*dictionary['param3']
    if function < 0:
        return 9999999
    return function

vector = pd.DataFrame(columns=['param1', 'param2', 'param3'])
vector['data'] = ['mu', 'std', 'min', 'max']
vector = vector.set_index('data')
vector.loc['mu'] = [5, 8, 1]
vector.loc['std'] = 20
vector.loc['min'] = 0
vector.loc['max'] = 100

EDA = EDAc(SIZE_GEN=40, MAX_ITER=200, DEAD_ITER=20, ALPHA=0.7, vector=vector, 
            aim='minimize', cost_function=cost_function)
bestcost, params, history = EDAc.run()
print(bestcost, params)
print(history)
```

In this case, the aim is to optimize a cost function which we want to minimize. The three parameters to optimize are continuous. This EDA must be initialized with some initial values (mu), and an initial range to search (std). Optionally, a minimum and a maximum can be specified.

As in the binary EDA, the best cost found, the solution and the cost evolution is returned.

#### Continuous multivariate EDA

In this case, dependencies among the variables are considered and managed with a Gaussian Bayesian Network. This EDA must be initialized with an historical in order to try to find the optimum based on it. A parameter (beta) is included to control the influence of the historic in the final solution. Some of the variables can be evidences (fixed values for which we want to find the optimum of the other variables). 
The optimizer will find the optimum values of the non-evidence-variables based on the value of the evidences. This is widely used in problems where dependencies among variables must be considered.

```python
from EDApy import EDA_multivariate as EDAm
import pandas as pd

blacklist = pd.DataFrame(columns=['from', 'to'])
aux = {'from': 'param1', 'to': 'param2'}
blacklist = blacklist.append(aux, ignore_index=True)

data = pd.read_csv(path_CSV)  # columns param1 ... param5
evidences = [['param1', 2.0],['param5', 6.9]]

def cost_function(dictionary):
    return dictionary['param1'] + dictionary['param2'] + dictionary['param3'] + dictionary['param4'] + dictionary['param5']

EDAm = EDAm(MAX_ITER=200, DEAD_ITER=20, data=data, ALPHA=0.7, BETA=0.4, cost_function=cost_function,
                 evidences=evidences, black_list=blacklist, n_clusters=6, cluster_vars=['param1', 'param5'])
output = EDAm.run(output=True)

print('BEST', output.best_cost_global)
```
This is the most complex EDA implemented. Bayesian networks are used to represent an abstraction of the search space of each iteration, where new individuals are sampled. As a graph is a representation with nodes and arcs, some arcs can be forbidden by the black list (pandas dataframe with the forbidden arcs). 

In this case the cost function is a simple sum of the parameters. The evidences are variables that have fixed values and are not optimized. In this problem, the output would be the optimum value of the parameters which are not in the evidences list.
Due to the evidences, to help the structure learning algorithm to find the arcs, a clustering by the similar values is implemented. Thus, the number of clusters is an input, as well as the variables that are considered in the clustering.

In this case, the output is the self class that can be saved as a pickle in order to explore the attributes. One of the attributes is the optimum structure of the optimum generation, from which the structure can be plotted and observe the dependencies among the variables. Function the plot the structure is the following:
```python
from EDApy import print_structure
print_structure(structure=structure, var2optimize=['param2', 'param3', 'param4'], evidences=['param1', 'param5'])
```
![Structure praph plot](structure.png "Structure of the optimum generation found by the EDA")

## Getting started

#### Prerequisites
R must be installed to use the multivariate EDA, with installed libraries c("bnlearn", "dbnR", "data.table")
To manage R from python, rpy2 package must be also installed.

#### Installing

