import requests
from pyspark import SparkContext
from pyspark import SparkConf
from pyspark.sql.session import SparkSession
from pyspark.sql import Row
from pyspark.sql.functions import udf
from pyspark.ml.feature import CountVectorizer
import re
from pyspark.sql.types import ArrayType, StringType
import pickle
from pyspark.ml.feature import VectorAssembler
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.tree import GradientBoostedTrees
from pyspark.mllib.tree import RandomForest
import pandas as pd
import pandas
import argparse
import sys

def get_file_names(file):
    '''Takes in the file name text file (as a Response object).
    Creates a list of the file names from the Response object.
    '''
    file_names = []
    for line in file.iter_lines():
        line = line.decode(errors='ignore')
        file_names.append(line)
    return file_names

def get_features(train_path, test_path, features_used):
    '''Extracts counts of bytes from byte files and
    counts of headers from asm files.
    '''
    
    #Get file name text file as a Response object.
    train_file = requests.get(train_path, stream=True)
    test_file = requests.get(test_path, stream=True)
    
    #Get a list of file names from the Response object.
    train_file_names = get_file_names(train_file)
    test_file_names = get_file_names(test_file)
    
    train_len = len(train_file_names)
    test_len = len(test_file_names)


    #Creates a 2D list of length equal to number of files.
    #Each element is a file's feature list.
    #Each element is initally set to a large number of 0s. 
    #We can pass any number thought to be larger than expected number of features.
    X_train = [[0]*2000 for x in range(train_len)]
    X_test = [[0]*2000 for x in range(test_len)]
    
    #Index corresponds to the index of the current feature.
    #If a new feature is seen, that feature's index is set to the current value of index...
    #As no features are seen yet, we set it to 0.
    index = 0 
    
    #Appeared contains all the features that have previously appeared with their corresponding indices.
    #It makes sure that the feature list of all the files are in the same order.
    #For ex. if the first three features seen are '00', 'FB', '11'...
    #Then appeared['00'] = 0, appeared['FB'] = 1 and appeared['11'] = 2
    #Then if any file has a feature list of = [10, 30, 20.......], that means it has...
    #... 10 appearences of '00', 30 of 'FB', 20 of '11' and so on.
    appeared = {}
    
    train_flag = 0
    test_flag = 1

    #Here we extract the byte count features. 
    #The newly seen features are stored in the appeared dictionary, with their corresponding indices.
    #If train_flag is passed, then newly seen features are added to the appeared dictionary.
    #If test_flag is passed, then newly seen features are ignored.
    if features_used[0] == '1':
        X_train, index, appeared = get_byte_features(X_train[:], train_file_names, index, appeared, train_flag)
        X_test, index, appeared = get_byte_features(X_test[:], test_file_names, index, appeared, test_flag)
    
    #Here we extract the asm count features.
    if features_used[1] == '1':
        X_train, index, appeared = get_asm_features(X_train[:], train_file_names, index, appeared, train_flag)
        X_test, index, appeared = get_asm_features(X_test[:], test_file_names, index, appeared, test_flag)
    
    #Here we join the byte and asm features for each file.
    #We must also remove all the extra 0s which were added during initialization. 
    #The current value of index is the index of the last feature + 1.
    #So we slice each files' list to end at index. This removes all the extra 0s. 

    X_train = [x[:index] for x in X_train[:]]
    X_test = [x[:index] for x in X_test[:]]
    
    #Saving features to disk for future.
    with open('X_train_new', 'wb') as fp:
        pickle.dump(X_train, fp)
        
    with open('X_test_new', 'wb') as fp:
        pickle.dump(X_test, fp)
        
    
    return X_train, X_test
    
def get_byte_features(X, file_list, index, appeared, test_flag):
    '''This function extracts the byte counts of each byte file.
    '''
    for file_index, f in enumerate(file_list):
        print(file_index)
        
        #Create a Response object of a particular byte file.
        r = requests.get("https://storage.googleapis.com/uga-dsp/project1/data/bytes/" + f +".bytes", stream=True)
        #Creating a dictionary with bytes as KEYS and their counts as VALUES.
        #Read the byte file line by line.
        for line in r.iter_lines():
            line = line.decode(errors='ignore')
            if len(line) == 0:
                continue
            #Splitting the line on whitespaces and looping over each byte.
            line = line.split()
            
            for byte in line[1:]:
                #Ignoring bytes with ? character.
                if '?' in byte:
                    continue
                else:
                    #Checking if the byte has appeared before in any file.
                    #If not, adding it to the appeared dictionary (only for training data).
                    #Otherwise, incrementing its count by 1.
                    if byte not in appeared:
                        #Does not add new features from the test set.
                        if test_flag == 1:
                            continue
                        else:
                            #Adding newly seen bytes to the appeared dictionary.
                            #The newly added byte's index is set to the current value of index.
                            #Incrementing index for the next feature.
                            appeared[byte] = index
                            X[file_index][index] += 1
                            index += 1
                    else:
                        #For an already seen byte, we get that byte's index.
                        #And then increment the value at that index.
                        X[file_index][appeared[byte]] += 1

    return X, index, appeared

def get_asm_features(X, file_list, index, appeared, test_flag):
    '''This function extracts the header counts of each asm file.
    It returns a list of headers that appeared across all the files and...
    ...a list of header counts for each file.
    '''
    for file_index, f in enumerate(file_list):
        print(file_index)
        #Create a Response object of a particular byte file.
        r = requests.get("https://storage.googleapis.com/uga-dsp/project1/data/asm/" + f +".asm", stream=True)
        for line in r.iter_lines():
            #Read the asm file line by line.
            line = line.decode(errors='ignore')
            if len(line) == 0:
                continue
            #Splitting the line on ':' and selecting the first element.
            header = line.split(':')[0]
            #Checking if the header has appeared before in any file.
            #If not, adding it to the appeared dictionary (only for training data).
            #Otherwise, incrementing its count by 1.
            if header not in appeared:
                if test_flag == 1:
                    continue
                else:
                    #Adding newly seen headers to the appeared dictionary.
                    #The newly added header's index is set to the current value of index.
                    #Incrementing index for the next feature.
                    appeared[header] = index
                    X[file_index][index] += 1
                    index += 1
            #For an already seen header, we get that header's index
            #And then increment the value at that index.
            else:
                X[file_index][appeared[header]] += 1

    return X, index, appeared

def get_labels(y_path):
    '''Creates a list of labels for training.
    '''
    y_train_file = requests.get(y_path, stream=True)
    y_train = []
    for line in y_train_file.iter_lines():
        line = line.decode(errors='ignore')
        #1 is subtracted because spark requires labels start from 0.
        y_train.append(int(line) - 1)
    return y_train

def create_training_matrix(X_train, y_train):
    '''Creates a training matrix to be passed to spark.
    '''
    data = []
    train_len = len(X_train)    
    for i in range(train_len):
        data.append(LabeledPoint(y_train[i], X_train[i]))
    return data

def predict_and_save(sc, train_data, X_test, num_trees_range=[40, 41], max_depth_range=[30, 31]):
    '''Trains a Random Forest classifier.
    Loops over different values of trees and max depth.
    '''
    for trees in range(num_trees_range[0], num_trees_range[1]):
        for depth in range(max_depth_range[0], max_depth_range[1]):
            model = RandomForest.trainClassifier(sc.parallelize(train_data), 9, {}, trees, maxDepth = depth)
            a = []
            for i in X_test:
                 a.append(int(model.predict(i) + 1))
            trees_s = str(trees)
            depth_s = str(depth)
            string = 'submit' + trees_s + depth_s + '.txt'
            b = pd.DataFrame(a)
            b.to_csv(string, header = False, index = False)


def set_parameters(arg_list):
    '''Selects the dataset and features used. 
    Defaults to large dataset and all features used.
    '''
    print(sys.argv)
    dataset ='l'
    features_used = '11'
    if len(sys.argv) >= 2:
        dataset = sys.argv[1]
        if dataset != 'l' and dataset != 's':
            dataset = 'l'
    if len(sys.argv) >= 3:
        features_used = sys.argv[2]
        if features_used != '00' and features_used != '01' and features_used != '10' and features_used != '11':
            features_used = '11'
    return dataset, features_used

def main():

    dataset, features_used = set_parameters(sys.argv)


    #Default to use all features in case of invalid parameters.


    #Setting parameters for Spark
    sc = SparkContext.getOrCreate()
    sc.stop()
    conf = SparkConf().setAll([('spark.executor.memory', '7000M'), ('spark.rpc.message.maxSize', '2047'), ('spark.ui.showConsoleProgress', 'true')])
    sc = SparkContext.getOrCreate(conf=conf)
    print(sc.getConf().getAll())
    spark = SparkSession(sc)
    
    #Settin URls for files
    if dataset == 'l':
        X_train_path = 'https://storage.googleapis.com/uga-dsp/project1/files/X_train.txt'
        X_test_path = 'https://storage.googleapis.com/uga-dsp/project1/files/X_test.txt'
        y_path = 'https://storage.googleapis.com/uga-dsp/project1/files/y_train.txt'
    else:
        X_train_path = 'https://storage.googleapis.com/uga-dsp/project1/files/X_small_train.txt'
        X_test_path = 'https://storage.googleapis.com/uga-dsp/project1/files/X_small_test.txt'
        y_path = 'https://storage.googleapis.com/uga-dsp/project1/files/y_small_train.txt'
    print(X_train_path)
    #Creating a list of y labels
    y_train = get_labels(y_path)
    
    #Extracing features from train and test set
    X_train, X_test = get_features(X_train_path, X_test_path, features_used)
    
    #Creating a training matrix by combing features and labels
    train_data = create_training_matrix(X_train, y_train)
    
    #Passing training data to the model and outputing prediction
    predict_and_save(sc, train_data, X_test)


if __name__ == "__main__":
    main()
