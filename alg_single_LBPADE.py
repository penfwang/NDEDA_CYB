from __future__ import division
import bisect
import math
import random
from itertools import chain
from operator import attrgetter, itemgetter
from collections import defaultdict
import numpy as np
from deap import base


toolbox = base.Toolbox()

def euclidean_distance(x1,x2):
    s1 = toolbox.clone(x1)
    s2 = toolbox.clone(x2)
    s1 = np.array(s1)
    s2 = np.array(s2)
    temp = sum((s1-s2)**2)
    temp1 = np.sqrt(temp)
    return temp1

def find_all(a_str, sub):
    start = 0
    while True:
        start = a_str.find(sub, start)
        if start == -1: return
        yield start
        start += len(sub)



def findindex(org, x):
    result = []
    for k,v in enumerate(org): #k和v分别表示org中的下标和该下标对应的元素
        if v == x:
            result.append(k)
    return result



######################################
#def selNS(individuals, k):
   # if len(individuals) < k:
   #     return individuals
   # pop_fit = [ind.fitness.value for ind in individuals]
   # sorting = np.argsort(pop_fit)####sorting from minimum to maximum.
  #  pop = [individuals[t] for t in sorting[:k]]
  #  return pop



def selNS(pop,k,ee):###################################store the solution who have the same classifiction error
    if len(pop) == k:
        return pop
    pop_fit = np.array([ind.fitness.value for ind in pop])
    index = np.argsort(pop_fit)################sortings' index
    fit_sort = sorted(pop_fit)##fitness' sorting
    # print('index',index)
    # print('fit_sort',fit_sort)
    if abs(fit_sort[k - 1] - fit_sort[k]) <= ee:####
         have_preserve_length = len(np.argwhere((fit_sort[k - 1]- fit_sort) > ee))
         # print('index[:have_preserve_length]',index[:have_preserve_length],have_preserve_length)
         off = [pop[m1] for m1 in index[:have_preserve_length]]
         need_more_length = k-have_preserve_length
         # print('need_more_length',need_more_length)
         index_fitness = np.argwhere(abs(fit_sort - fit_sort[k - 1]) <= ee)
         # print(index_fitness)
         list1 = []
         for ii in index_fitness:
             iii = random.choice(ii)
             list1.append(iii)
         list2 = [index[t] for t in list1]####from list2 choose need_more_length solutions to off
         # print(list2)
         # exit()
         pop_list2 = [pop[m2] for m2 in list2]
         size_solutions_in_pop_list2 = obtain_size(pop_list2)
         index1 = np.argsort(size_solutions_in_pop_list2)  ################sortings' index
         need_index = index1[:need_more_length]
         need_save = [list2[m3] for m3 in need_index]
         [off.append(pop[m4]) for m4 in need_save]
    else:
        off = [pop[k] for k in index[:k]]
    return off


def obtain_size(pop):
    size = []
    for z in pop:
       tt_01 = 1 * (np.array(z) >= 0.6)
       tt_01 = "".join(map(str, tt_01))  ######## the '0101001' of the current individual
       z_index = np.array(list(find_all(tt_01, '1')))  ##### the selected features of the current individual
       size.append(len(z_index))
    return size




def selection_compared_with_nearest(old,new):
    pop = toolbox.clone(new)
    dis1 = np.zeros((len(new),len(old)))
    for i in range(len(new)):
        for j in range(len(old)):
            dis1[i, j] = euclidean_distance(new[i], old[j])
        min_index = np.argwhere(dis1[i, :] == min(dis1[i, :]))
        min_one = random.choice(min_index)
        min_one = random.choice(min_one)
        temp = new[i].fitness.value-old[min_one].fitness.value
        if temp>0:
            pop[i] = old[min_one]
        else:
            pop[i] = new[i]
    return pop
