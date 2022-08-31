from __future__ import division
import bisect
import math
import random
from itertools import chain
from operator import attrgetter, itemgetter
from collections import defaultdict
import numpy as np
import itertools
from minepy import MINE
from deap import tools, base
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
toolbox = base.Toolbox()



def fit_train(x1, train_data):
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
     f2 = (len(value_position)-1)/(train_data.shape[1] - 1)
     # f2 = len(value_position) - 1
    return f1, f2


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


def obtain_r(z, x_train, mic_value):
    tt_01 = 1 * (np.array(z) >= 0.6)
    tt_01 = "".join(map(str, tt_01))  ######## the '0101001' of the current individual
    z_index = np.array(list(find_all(tt_01, '1')))  ##### the selected features of the current individual
    r_cf = mic_value[z_index]  ########## the related mic of the features with label
    average_r_cf = np.mean(r_cf)
    aa = list(itertools.combinations(z_index, 2))### the unique combinations between pair of features
    mine = MINE(alpha=0.6, c=15)
    mic_ff = []
    for i_in in range(len(aa)):
           mine.compute_score(x_train[:, aa[i_in][0] + 1], x_train[:, aa[i_in][1] + 1])
           mic_ff.append(mine.mic())
    mic_ff = np.array(mic_ff)
    average_r_ff = np.mean(mic_ff)
    k =len(tt_01)
    m_value = (k * average_r_cf) / math.sqrt(k + k * (k - 1) * average_r_ff)
    return m_value

def ran(number):
    i = 0
    y = np.zeros((number))
    while i< number:
        y[i]=random.random()
        i = i+1
    return y


def add_delete(temp,p_add,p_delete,dim):
    y_add = random.random()
    y_delete = random.random()
    inter = random.randint(0,dim-1)
    if y_add < p_add[inter]:
            temp[inter] = 1
    if y_delete < p_delete[inter]:
            # temp[t2] = random.uniform(0, 0.6)
            temp[inter] = 0
    return temp


def get_whole_01(individuals):
    all_index = []
    individuals_array = np.array(individuals)  ####
    for i0 in range(individuals_array.shape[0]):
        x1 = 1 * (individuals_array[i0, :] >= 0.6)
        x1 = "".join(map(str, x1))  # transfer the array form to string in order to find the position of 1
        all_index.append(x1)  ##store all individuals who have changed to 0 or 1
    return all_index



def mutDE(a, b, c,d):###mutation:DE/rand/1
    f = 0.9
    y = toolbox.clone(a)
    for i in range(len(y)):
        # y[i] = a[i] + f*(b[i]-c[i])
        c1 = random.random()
        c2 = random.random()
        for i in range(len(y)):
            #y[i] = y[i] + c1 * (b[i] - y[i]) + c2 * (c[i] - d[i])
            y[i] = y[i] + c1 * (b[i] - y[i]) + c[i] - d[i]
    return y



def DE_mutation(temp,pop_non):
    if len(pop_non) == 0:
        temp_new = temp
    elif len(pop_non) == 1:
        index = [0,0]
        b = pop_non[index[0]]
        c = pop_non[index[1]]
        temp_new = mutDE(temp, b, c)
    else:
      b,c,d = random.sample(pop_non, 3)
      temp_new = mutDE(temp, b, c,d)
    return temp_new


def whether_shown_before(a,all):
    b = np.array(a)
    x1 = 1 * (b >= 0.6)
    x1 = "".join(map(str, x1))
    state = findindex(all, x1)
    return state




def random_mutation(old):
    old1 = np.array(old)
    old1 = 1 * (old1 >= 0.6)
    old1 = "".join(map(str, old1))
    select_position = np.array(list(find_all(old1, '1')))
    non_position = np.array(list(find_all(old1, '0')))
    swap = min(int(0.5*len(select_position)),len(non_position))
    if swap == 0:
        swap = 1
    num = random.randint(1,swap)####randomly produce a number that means the number of changes
    temp1 = random.sample(list(select_position),num)
    temp2 = random.sample(list(non_position),num)
    for i in temp1:
        old[i] = random.uniform(0,0.6)
    for j in temp2:
        old[j] =random.uniform(0.6,1)
    return old



def produce_unique_individuals(new,pop,dis,pop_unique,kk):
    if len(new) == 0:
        return
    unique_matrix_01 = get_whole_01(pop_unique)
    for i in range(len(new)):
        state = whether_shown_before(new[i],unique_matrix_01)
        count = 0
        if state != []:###shown before
            while count < 2:
                demo = new[i]
                new_solution = random_mutation(demo)
                #new_solution = DE_mutation(demo,niche_offspring)  ###from pop_non choose solution to mutate
                del demo
                for i_z in range(len(new_solution)):
                    if new_solution[i_z] > 1:
                        new_solution[i_z] = 1
                    if new_solution[i_z] < 0:
                        new_solution[i_z] = 0
                #demo = new_solution
                new_one1 = np.array(new_solution)
                new_one_011 = 1 * (new_one1 >= 0.6)
                new_one_011 = "".join(map(str, new_one_011))
                temp = findindex(unique_matrix_01, new_one_011)
                if len(temp) == 0:
                    new[i] = new_solution
                    pop_unique.append(new_solution)
                    unique_matrix_01 = get_whole_01(pop_unique)
                    break
                del temp
                count = count + 1
        else:
            pop_unique.append(new[i])
            unique_matrix_01 = get_whole_01(pop_unique)
 #####################################################思想是继续check是否已经存在，已经存在就继续直到生成新的unique解
    return new,pop_unique

def euclidean_distance(x1,x2):
    s1 = toolbox.clone(x1)
    s2 = toolbox.clone(x2)
    s1 = np.array(s1)
    s2 = np.array(s2)
    temp = sum((s1-s2)**2)
    temp1 = np.sqrt(temp)
    return temp1



def pre_selection(new,old,ee):
    pop=[]
    dis1 = np.zeros((len(new),len(old)))
    for i in range(len(new)):
        for j in range(len(old)):
            dis1[i, j] = euclidean_distance(new[i], old[j])
        min_index = np.argwhere(dis1[i, :] == min(dis1[i, :]))
        min_one = random.choice(min_index)
        min_one = random.choice(min_one)
        temp = new[i].fitness.value-old[min_one].fitness.value
        if abs(temp) <= ee:###need to preserve both
            pop.append(new[i])
            pop.append(old[min_one])
        else:
            if temp > 0:
               pop.append(old[min_one])
            else:
               pop.append(new[i])
    return pop
