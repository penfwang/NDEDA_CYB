import numpy as np
from deap import tools, base
toolbox = base.Toolbox()

def selNS(old,new):
    offspring = toolbox.clone(old)
    for i in range(len(old)):
        if old[i].fitness.values > new[i].fitness.values:
            offspring[i] = new[i]
        else:
            offspring[i] = old[i]
    return offspring