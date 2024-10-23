import numpy as np
import sys
import itertools
# you need to install this package `ioh`. Please see documentations here: 
# https://iohprofiler.github.io/IOHexp/ and
# https://pypi.org/project/ioh/
from ioh import get_problem, logger, ProblemClass

budget = 5000
dimension = 50
## tournament selection


# To make your results reproducible (not required by the assignment), you could set the random seed by
# `np.random.seed(some integer, e.g., 42)`

def s2399059_s2231085_GA(problem, problem_number):
    parents = []
    parent_fitness = []
    offspring = []
    f_optimal = 0
    if problem_number == 18:
        population_size = 15
        mutation_rate = 0.02
        crossover_p = 0.75
        tournament_percent = 0.25
    else:
        population_size = 20
        mutation_rate = 0.05
        crossover_p = 0.75
        tournament_percent = 0.5

    tournament_k = round(population_size*tournament_percent)
     
    # Initialization
    for i in range(population_size):
        parents.append(np.random.randint(2, size = dimension))
        parent_fitness.append(problem(parents[i])) #get fitness according to F problem

    while problem.state.evaluations < budget:
        offspring = []
        # Tournament Selection
        for i in range(len(parents)) :
            pre_select = np.random.choice(len(parent_fitness),tournament_k,replace = False)
            max_f = parent_fitness[pre_select[0]]
            index = pre_select[0]
            for p in pre_select:
                if parent_fitness[p] > max_f:
                    index = p
                    max_f = parent_fitness[p]
            offspring.append(parents[index].copy())

        # Crossover
        # uniform crossover for problem F19
        if problem_number == 19:   
            for i in range(len(offspring)):
                p1 = np.random.randint(0, len(offspring))
                p2 = np.random.randint(0, len(offspring))
                while p1 == p2:
                    p2 = np.random.randint(0, len(offspring))
                offspring_1 = offspring[p1]
                offspring_2 = offspring[p2]
                if(np.random.uniform(0,1) < crossover_p):
                    for i in range(len(offspring_1)):
                        if np.random.uniform(0,1) < 0.5:
                            offspring_1[i], offspring_2[i] = offspring_2[i], offspring_1[i]
              
        # 1-point crossover for problem F18
        if problem_number == 18: 
            for i in range(len(offspring)):
                p1 = np.random.randint(0, len(offspring))
                p2 = np.random.randint(0, len(offspring))
                while p1 == p2:
                    p2 = np.random.randint(0, len(offspring))
                offspring_1 = offspring[p1]
                offspring_2 = offspring[p2]
                crossover_point = np.random.randint(0,len(offspring[i]))
                for i in range(crossover_point, len(offspring_1)):
                    offspring_1[i], offspring_2[i] = offspring_2[i], offspring_1[i]
        
        # Mutation
        for i in range(len(offspring)):
            for j in range(len(offspring[i])):
                if np.random.uniform(0,1) < mutation_rate:
                    offspring[i][j] = 1-offspring[i][j]
  
        # Evaluation
        parents = offspring.copy()
        for i in range(population_size):
            parent_fitness[i] = problem(parents[i])
            if parent_fitness[i] > f_optimal:
                f_optimal = parent_fitness[i]
                x_optimal = parents[i].copy()

def create_problem(fid: int, problem_number: int):
    # Declaration of problems to be tested.
    problem = get_problem(fid, dimension=dimension, instance=1, problem_class=ProblemClass.PBO)

    # Create default logger compatible with IOHanalyzer
    # `root` indicates where the output files are stored.
    # `folder_name` is the name of the folder containing all output. You should compress the folder 'run' and upload it to IOHanalyzer.
    l = logger.Analyzer(
        root="data",  # the working directory in which a folder named `folder_name` (the next argument) will be created to store data
        folder_name="run",  # the folder name to which the raw performance data will be stored
        algorithm_name="genetic_algorithm_{}".format(problem_number),
        algorithm_info="Practical assignment of the EA course",
    )
    # attach the logger to the problem
    problem.attach_logger(l)
    return problem, l

if __name__ == "__main__":    
    # this how you run your algorithm with 20 repetitions/independent run
    F18, _logger = create_problem(18, 18)
    for run in range(20): 
        s2399059_s2231085_GA(F18,18)
        print("go")
        F18.reset() # it is necessary to reset the problem after each independent run
    _logger.close() # after all runs, it is necessary to close the logger to make sure all data are written to the folder

    F19, _logger = create_problem(19, 19)
    for run in range(20): 
        s2399059_s2231085_GA(F19,19)
        F19.reset()
    _logger.close()

# ============================================================================================================================================

# Uniform Crossover
def crossover(p1, p2):
   if(np.random.uniform(0,1) < crossover_probability):
    for i in range(len(p1)) :
        if np.random.uniform(0,1) < 0.5:
            t = p1[i]
            p1[i] = p2[i]
            p2[i] = t

# Standard bit mutation using mutation rate p
def mutation(p):
    for i in range(len(p)) :
        if np.random.uniform(0,1) < mutation_rate:
            p[i] = 1 - p[i]

# Using the Fitness proportional selection
def mating_seletion(parent, parent_f) :    
    # Using the tournament selection
    # select_parent = []
    # for i in range(len(parent)) :
    #     pre_select = np.random.choice(len(parent_f),tournament_k,replace = False)
    #     max_f = sys.float_info.min
    #     for p in pre_select:
    #         if parent_f[p] > max_f:
    #             index = p
    #             max_f = parent_f[p]
    #     select_parent.append(parent[index].copy())
    # return select_parent

    # Using the proportional selection

    # Plusing 0.001 to avoid dividing 0
    f_min = min(parent_f)
    f_sum = sum(parent_f) - (f_min - 0.001) * len(parent_f)
    
    rw = [(parent_f[0] - f_min + 0.001)/f_sum]
    for i in range(1,len(parent_f)):
        rw.append(rw[i-1] + (parent_f[i] - f_min + 0.001) / f_sum)
    
    select_parent = []
    for i in range(len(parent)) :
        r = np.random.uniform(0,1)
        index = 0
        # print(rw,r)
        while(r > rw[index]) :
            index = index + 1
        
        select_parent.append(parent[index].copy())
    return select_parent


def genetic_algorithm(func, budget = None):
    
    # budget of each run: 10000
    if budget is None:
        budget = 10000
    
    # f_opt : Optimal function value.
    # x_opt : Optimal solution.
    f_opt = sys.float_info.min
    x_opt = None
    
    # parent : A list that holds the binary strings representing potential solutions or individuals in the current population.
    # parent_f : A list that holds the fitness values corresponding to each individual in the parent list.
    parent = []
    parent_f = []
    for i in range(pop_size):

        # Initialization
        parent.append(np.random.randint(2, size = func.meta_data.n_variables))
        parent_f.append(func(parent[i]))
        budget = budget - 1

    while (f_opt < optimum and budget > 0):

        # Perform mating selection, crossover, and mutation to generate offspring
        # Dit is van werkcollege        
        offspring = mating_seletion(parent,parent_f)

        for i in range(0,pop_size - (pop_size%2),2) :
            crossover(offspring[i], offspring[i+1])


        for i in range(pop_size):
            mutation(offspring[i])

        parent = offspring.copy()
        for i in range(pop_size) : 
            parent_f[i] = func(parent[i])
            budget = budget - 1
            if parent_f[i] > f_opt:
                    f_opt = parent_f[i]
                    x_opt = parent[i].copy()
            if f_opt >= optimum:
                break
        
    # ioh function, to reset the recording status of the function.
    func.reset()
    print(f_opt,x_opt)
    return f_opt, x_opt

