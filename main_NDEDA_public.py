#This file is the single-objective LBPADE from the author
#try to ensemble the solutions
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)
simplefilter(action='ignore', category=UserWarning)
from itertools import chain
import array
import random
import json
import numpy as np
from math import sqrt
from deap import algorithms
from deap import base
import math,time
from deap import benchmarks
from deap import creator
from deap import tools
from operator import mul
from functools import reduce
import alg_single_LBPADE
import numpy.matlib
import operator
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score,cross_val_predict
from collections import Counter
import geatpy as ea
from single_diverse2 import produce_unique_individuals,pre_selection
import sys,saveFile


def uniform(low, up, size=None):####generate a matrix of the range of variables
    try:
        return [random.uniform(a, b) for a, b in zip(low, up)]
    except TypeError:
        return [random.uniform(a, b) for a, b in zip([low] * size, [up] * size)]

def findindex(org, x):
    result = []
    for k,v in enumerate(org): 
        if v == x:
            result.append(k)
    return result

def find_all(a_str, sub):
    start = 0
    while True:
        start = a_str.find(sub, start)
        if start == -1: return
        yield start
        start += len(sub)



def fit_train1(x1, train_data):
    x = np.zeros((1,len(x1)))
    for ii in range(len(x1)):
        x[0,ii] = x1[ii]
    x = random.choice(x)
    x = 1 * (x >= 0.6)
    if np.count_nonzero(x) == 0:
        f1 = 1###error_rate
        f2 = 1
    else:
     x = "".join(map(str, x))  # transfer the array form to string in order to find the position of 1
     value_position = np.array(list(find_all(x, '1'))) + 1  # cause the label in the first column in training data
     value_position = np.insert(value_position, 0, 0)  # insert the column of label
     tr = train_data[:, value_position]
     clf = KNeighborsClassifier(n_neighbors = 5)
     scores = cross_val_score(clf, tr[:,1:],tr[:,0], cv = 5)
     f1 = np.mean(1 - scores)
     f2 = len(value_position)-1
    f = f1 + 0.000001 * f2
    return f



def kNNClassify(newInput, dataSet, labels, k):
    numSamples = dataSet.shape[0]   
    diff = np.tile(newInput, (numSamples, 1)) - dataSet 
    squaredDiff = diff ** 2 
    squaredDist = squaredDiff.sum(axis = 1)   
    distance = squaredDist ** 0.5  
    sortedDistIndices = distance.argsort()
    classCount = {}
    for i in range(k):
        voteLabel = labels[sortedDistIndices[i]]
        classCount[voteLabel] = classCount.get(voteLabel, 0) + 1
    sorted_ClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_ClassCount[0][0]


def evaluate_test_data(x, train_data, test_data):
    x = 1 * (x >= 0.6)
    x = "".join(map(str, x))  # transfer the array form to string in order to find the position of 1
    value_position = np.array(list(find_all(x, '1'))) + 1  # cause the label in the first column in training data
    value_position = np.insert(value_position, 0, 0)  # insert the column of label
    te = test_data[:, value_position]#####testing data including label in the first colume
    tr = train_data[:, value_position]#####training data including label in the first colume too
    wrong = 0
    for i12 in range(len(te)):
        testX = te[i12,1:]
        dataSet = tr[:,1:]
        labels = tr[:,0]
        outputLabel = kNNClassify(testX, dataSet, labels, 5)
        # print(outputLabel,te[i12,0])
        if outputLabel != te[i12,0]:
            wrong = wrong + 1
    f1 = wrong/len(te)
    f2 = (len(value_position) - 1) / (test_data.shape[1] - 1)
    return f1, f2


def more_confidence(EXA, index_of_objectives):
    a = 0.6
    cr = np.zeros((len(index_of_objectives),1))
    for i in range(len(index_of_objectives)):###the number of indexes
        temp = 0
        object = EXA[index_of_objectives[i]]
        for ii in range(len(object)):###the number of features
           b = object[ii]
           if b > a:  con = (b - a) / (1 - a)
           else:      con = (a - b) / (a)
           temp = con + temp
        cr[i,0] = temp
    sorting = np.argsort(-cr[:,0])####sorting from maximum to minimum
    index_one = index_of_objectives[sorting[0]]
    return index_one


def delete_duplicate(EXA):####list
    EXA1 = []
    EXA_array = np.array(EXA)
    all_index = []
    for i0 in range(EXA_array.shape[0]):
       x = 1 * (EXA_array[i0,:] >= 0.6)
       x = "".join(map(str, x))  # transfer the array form to string in order to find the position of 1
       all_index.append(x)##store all individuals who have changed to 0 or 1
    single_index = set(all_index)####find the unique combination
    single_index = list(single_index)####translate it's form in order to following operating.
    for i1 in range(len(single_index)):
       index_of_objectives = findindex(all_index, single_index[i1])##find the index of each unique combination
       if len(index_of_objectives) == 1:
          for i2 in range(len(index_of_objectives)):
             EXA1.append(EXA[index_of_objectives[i2]])
       else:####some combination have more than one solutions.here may have duplicated solutions
           index_one = more_confidence(EXA, index_of_objectives)
           EXA1.append(EXA[index_one])
    # print(EXA1)
    return EXA1

def generate(size, pmin, pmax, smin, smax):
    part = creator.Particle(numpy.random.uniform(pmin, pmax, size))
    part.speed = numpy.random.uniform(smin, smax, size)
    part.smin = smin
    part.smax = smax
    return part


def mutDE(y, a, b, c, f):###mutation:DE/rand/1; if a is the best one, it will change as DE/best/1
    for i in range(len(y)):
        y[i] = a[i] + f*(b[i]-c[i])
    return y



def proposed_mutation2(dis,offspring,ii,nei,gen,fit_num,Max_FES):#########if best solution is larger than a half, select solutions from current niche
    ss = np.argsort(dis)
    niche_offspring = [offspring[t] for t in ss[1:nei]]###use to compare the fitness
    member = offspring[ii]
    s = [(member.fitness.value>=t.fitness.value)*1 + (member.fitness.value<t.fitness.value)*0 for t in niche_offspring]
    sum_abs = sum(s)
    #f_mu=(fit_num<=(0.8*Max_FES))*((sum_abs/(nei))*0.8+0.1)+(fit_num>(0.8*Max_FES))*(((sum_abs/(nei))* 0.8+0.1)*0.001)
    f_mu= 0.5
    pop_fit_whole = [ind.fitness.value for ind in offspring]
    niche_offspring1 = [offspring[t] for t in ss[:nei]]
    pop_fit = [ind.fitness.value for ind in niche_offspring1]
    if sum_abs < 4:######from whole population choose nbest, while from niche randomly choose other two solutions
        min_index = np.argwhere(pop_fit_whole == min(pop_fit_whole))
        min_one = random.choice(min_index)
        min_one = random.choice(min_one)
        nbest = offspring[min_one]
        offspring1 = [niche_offspring1[t] for t in range(len(niche_offspring1))]  ###use to compare the fitness
    else:##from niche choose solution as nbest
        min_index = np.argwhere(pop_fit == min(pop_fit))
        min_one = random.choice(min_index)
        min_one = random.choice(min_one)
        nbest = niche_offspring1[min_one]
        offspring1 = [offspring[t] for t in range(len(offspring))]  ###use to compare the fitness
    if member in offspring1:
            offspring1.remove(member)
    if nbest in offspring1:
            offspring1.remove(nbest)
    in1, in2 = random.sample(offspring1, 2)
    y_new = toolbox.clone(member)
    for i2 in range(len(y_new)):
               y_new[i2] = member[i2] + f_mu * (nbest[i2] - member[i2])+ f_mu  * (in1[i2] - in2[i2])
    return y_new, nbest


##cxBinomial(offspring[ii],y_new,0.5)###crossover
def cxBinomial(x, y, cr):#####binary crossover
    y_new = toolbox.clone(y)
    size = len(x)
    index = random.randrange(size)
    for i in range(size):
        if i == index or random.uniform(0, 1) <= cr:
            y_new[i] = y[i]
            # y_new[i] = 0.8*(1- y[i])
        else:
            y_new[i] = x[i]
    return y_new



def continus2binary(x):
    for i in range(len(x)):
            if x[i] >= 0.6:
                x[i] = 1.0
            else:
                x[i] = 0.0
    return x


def hamming_distance(s, s0):
    """Return the Hamming distance between equal-length sequences"""
    s1 = toolbox.clone(s)
    s2 = toolbox.clone(s0)
    s3 = continus2binary(s1)
    s4 = continus2binary(s2)
    if len(s3) != len(s4):
        raise ValueError("Undefined for sequences of unequal length")
    return sum(el1 != el2 for el1, el2 in zip(s3, s4))


def euclidean_distance(x1,x2):
    s1 = toolbox.clone(x1)
    s2 = toolbox.clone(x2)
    s1 = np.array(s1)
    s2 = np.array(s2)
    temp = sum((s1-s2)**2)
    temp1 = np.sqrt(temp)
    return temp1



def xor(a,b):
    xor_value = (1-a)*b+ a*(1-b)
    return xor_value


creator.create("FitnessMin", base.Fitness, weights=(-1.0,))####minimise an objectives
creator.create("Individual", array.array, typecode='d', fitness=creator.FitnessMin)
toolbox = base.Toolbox()

def main(seed,x_train):
    random.seed(seed)
    NDIM = x_train.shape[1] - 1
    ee = 1/x_train.shape[0]
    BOUND_LOW, BOUND_UP = 0.0, 1.0
    NGEN = 100###the number of generation
    if NDIM < 300:
        MU = NDIM ####the number of particle
    else:
        MU = 300  #####bound to 300
    Max_FES = MU * 100
    nei = 9
    min_fitness = []
    unique_number = []
    # toolbox.register("attr_float", bytes, BOUND_LOW, BOUND_UP, NDIM)
    toolbox.register("attr_float", uniform, BOUND_LOW, BOUND_UP, NDIM)  #####dertemine the way of randomly generation and gunrantuu the range
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_float)  ###fitness
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)  ##particles
    toolbox.register("evaluate", fit_train1, train_data= x_train)
    toolbox.register("select", alg_single_LBPADE.selNS)##alg4_NSGA2
    toolbox.register("select1", alg_single_LBPADE.selection_compared_with_nearest) 
    offspring = toolbox.population(n=MU)
    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)#####toolbox.evaluate = fit_train
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.value = fit
    pop_fit = [ind.fitness.value for ind in offspring]
    min_fitness.append(min(pop_fit))
    fit_num = len(offspring)
    #offspring = toolbox.select(offspring, len(offspring))
    pop_surrogate = delete_duplicate(offspring)
    unique_number.append(len(pop_surrogate))
    dis = np.zeros((MU,MU))
    for i in range(MU):
        for j in range(MU):
            dis[i,j] = hamming_distance(offspring[i],offspring[j])/NDIM
            #dis[i, j] = euclidean_distance(offspring[i], offspring[j]) / NDIM
    for gen in range(1, NGEN):
        pop_new = toolbox.clone(offspring)
        for ii in range(len(offspring)):#####upate the whole population
            y_new,nbest= proposed_mutation2(dis[ii, :],offspring,ii,nei,gen,fit_num,Max_FES)
            for i_z in range(len(y_new)):
                    if y_new[i_z] > 1:
                        y_new[i_z] = nbest[i_z]
                    if y_new[i_z] < 0:
                        y_new[i_z] = nbest[i_z]
            ss = np.argsort(dis[ii, :])
            pop_new[ii] = cxBinomial(offspring[ii],y_new,0.5)###crossover
            del pop_new[ii].fitness.values###delete the fitness
        # pop_new = produce_diverse_individuals(pop_new, pop_non)
        ##################################################
        pop_unique = delete_duplicate(offspring)
        if NDIM <= 500:
           pop_new,pop_unique = produce_unique_individuals(pop_new,offspring,dis,pop_unique,nei)
        pop_surrogate.extend(pop_unique)
        pop_surrogate = delete_duplicate(pop_surrogate)
        unique_number.append(len(pop_surrogate))
        invalid_ind = [ind for ind in pop_new if not ind.fitness.valid]
        fitne = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit1 in zip(invalid_ind, fitne):
            ind.fitness.value = fit1
        # Select the next generation population
        fit_num = fit_num + len(offspring)
        pop_mi = pop_new + offspring
        pop1 = delete_duplicate(pop_mi)
        offspring = toolbox.select(pop1, MU,ee)
        #offspring = toolbox.select1(offspring, pop_new)
        pop_fit = [ind.fitness.value for ind in offspring]######selection from author
        min_fitness.append(min(pop_fit))
        # offspring = toolbox.select(pop_new + offspring, MU)
        for i in range(MU):
            for j in range(MU):
                dis[i, j] = hamming_distance(offspring[i], offspring[j]) / NDIM
                #dis[i, j] = euclidean_distance(offspring[i], offspring[j]) / NDIM
        ##########################################################new
        if fit_num > Max_FES:
            break
    return offspring,min_fitness,unique_number


if __name__ == "__main__":
    dataset_name = str(sys.argv[1])
    seed = str(sys.argv[2])
    folder1 =  '/nfs/home/wangpe/split_73' + '/' + 'train' + str(dataset_name) + ".npy"
    #folder2 = '/nfs/home/wangpe/split_73' + '/' + 'test' + str(dataset_name) + ".npy"
    #x_test = np.load(folder2)
    x_train = np.load(folder1)
    start = time.time()
    pop,min_fitness,unique_number = main(seed,x_train)
    end = time.time()
    running_time = end - start
    pop1 = delete_duplicate(pop)
    pop_fit = [ind.fitness.value for ind in pop1]
    EXA_array = np.array(pop1)
    saveFile.saveAllfeature2(seed, dataset_name, EXA_array)
    saveFile.saveAllfeature3(seed, dataset_name, pop_fit)
    saveFile.saveAllfeature5(seed, dataset_name, unique_number)
    saveFile.saveAllfeature6(seed, dataset_name, min_fitness)
    saveFile.saveAllfeature7(seed, dataset_name, running_time)
