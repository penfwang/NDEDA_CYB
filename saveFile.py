import pickle
import numpy as np

def saveResults(fileName, *args, **kwargs):
    f = open(fileName, 'w')#####Truncate file to zero length or create text file for writing
    for i in args:
        f.writelines(str(i)+'\n')
    f.close()
    return

def saveLog (fileName, log):
    f=open(fileName, 'wb')###open the file with binary model for writing
    pickle.dump(log, f)
    f.close()
    return

#def saveAllResults(randomSeeds, dataSetName, running_time,train_accuracy,test_accuracy,whole_accuracy, frequency_accuracy, tempEXA,
# real_value_features,value_features_01):####store the time:
def saveAlltime(randomSeeds, dataSetName, running_30):
    fileName1='running_30' + dataSetName + '.txt'
    saveResults(fileName1, running_30)
    return


def saveAlltrain(randomSeeds, dataSetName, train_accuracy_30):
    fileName1= str(randomSeeds)+'train_accuracy_30' + dataSetName + '.txt'
    saveResults(fileName1, 'train', train_accuracy_30)
    return    

def saveAllrandomly(randomSeeds, dataSetName, randomly_test_accuracy_30):
    fileName1= str(randomSeeds)+'randomly_test_accuracy_30' + dataSetName + '.txt'
    saveResults(fileName1, 'randomly_test', randomly_test_accuracy_30)
    return 
def saveAllfrequency(randomSeeds, dataSetName, frequency_accuracy_30):
    fileName1= str(randomSeeds)+'frequency_accuracy_30' + dataSetName + '.txt'
    saveResults(fileName1, 'frequency', frequency_accuracy_30)
    return    

def saveAllwhole(randomSeeds, dataSetName, whole_accuracy_30):
    fileName1= str(randomSeeds)+'whole_accuracy_30' + dataSetName + '.txt'
    saveResults(fileName1, 'whole', whole_accuracy_30)
    return 
def saveAllfeature(randomSeeds, dataSetName, value_features_01):
    fileName1= str(randomSeeds)+'value_features_01_30' + dataSetName + '.txt'
    saveResults(fileName1, 'feature', value_features_01)
    return    

def saveAllfeature1(randomSeeds, dataSetName, EXA_01):
    fileName1= str(randomSeeds)+'EXA_01' + dataSetName
    np.save(fileName1, EXA_01)
    return

def saveAllfeature2(randomSeeds, dataSetName, EXA_array):
    fileName1= str(randomSeeds)+'EXA_array' + dataSetName
    np.save(fileName1, EXA_array)
    return

def saveAllfeature3(randomSeeds, dataSetName, front_training):
    fileName1= str(randomSeeds)+'front_training' + dataSetName
    np.save(fileName1, front_training)
    return
def saveAllfeature4(randomSeeds, dataSetName, front_testing):
    fileName1= str(randomSeeds)+'front_testing' + dataSetName
    np.save(fileName1, front_testing)
    return

def saveAllfeature5(randomSeeds, dataSetName, unique_number):
    fileName1= str(randomSeeds)+'unique_number' + dataSetName
    np.save(fileName1, unique_number)
    return

def saveAllfeature6(randomSeeds, dataSetName, min_fitness):
    fileName1= str(randomSeeds)+'min_fitness' + dataSetName
    np.save(fileName1, min_fitness)
    return

def saveAllfeature7(randomSeeds, dataSetName, running_time):
    fileName1= str(randomSeeds)+'running_time' + dataSetName
    np.save(fileName1, running_time)
    return

def saveAllfeature8(randomSeeds, dataSetName, ensemble_solution):
    fileName1= str(randomSeeds)+'ensemble_solution' + dataSetName
    np.save(fileName1, ensemble_solution)
    return


def saveAllfeature9(randomSeeds, dataSetName, spaces):
    fileName1= str(randomSeeds)+'spaces' + dataSetName
    np.save(fileName1, spaces)
    return

def saveAllhyp1(dataSetName, hyp_30_training):
    fileName1= 'hyp_30_training' + dataSetName + '.txt'
    saveResults(fileName1, hyp_30_training)
    return

def saveAllhyp2(dataSetName, hyp_30_testing):
    fileName1= 'hyp_30_testing' + dataSetName + '.txt'
    saveResults(fileName1, hyp_30_testing)
    return
