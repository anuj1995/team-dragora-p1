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


def get_file_names(file):
    """Takes in the file name text file (as a Response object).
    Creates a list of the file names from the Response object.
    """
    file_names = []
    for line in file.iter_lines():
        line = line.decode(errors='ignore')
        file_names.append(line)
    return file_names

def get_features(train_path, test_path):
    """Extracts counts of bytes from byte files and
    counts of headers from asm files.
    """
    
    #Get file name text file as a Response object.
    train_file = requests.get(train_path, stream=True)
    test_file = requests.get(test_path, stream=True)
    
    #Get a list of file names from the Response object.
    train_file_names = get_file_names(train_file)
    test_file_name = get_file_names(test_file)

    #all_byte_list_train/test is a 1-D list which contains all the bytes that appear across the train/test data.
    #byte_each_file_list_train/test is a 1-D list in which each element corresponds to one file...
    #...Each element is a dictionary with bytes as the KEY and counts of that byte as the VALUE.
    all_byte_list_train, byte_each_file_train = get_byte_features(train_file_names)
    all_byte_list_test, byte_each_file_test = get_byte_features(test_file_name)
    
    #Combining all the bytes appearing across the train and test set and removing duplicates.
    #A combined list is required because it is possible that some bytes only appear in the test set.
    all_byte_list = all_byte_list_train + all_byte_list_test
    all_byte_list = list(set(all_byte_list))
    
    #X_byte_train/test is a 2D list. Each element corresponds to one file...
    #...Each element is a list of the counts of the bytes appearing in that file.
    #Bytes that do not appear in a file are given a count of 0.
    X_byte_train = prepare_feature_matrix(all_byte_list, byte_each_file_train)
    X_byte_test = prepare_feature_matrix(all_byte_list, byte_each_file_test)
    
    #all_asm_list_train/test is a 1-D list which contains all the headers that appear across the train/test data.
    #asm_each_file_list_train/test is a 1-D list in which each element corresponds to one file...
    #...Each element is a dictionary with headers as the KEY and counts of that header as the VALUE.
    all_asm_list_train, asm_each_file_train = get_asm_features(train_file_names)
    all_asm_list_test, asm_each_file_test = get_asm_features(test_file_name)

    #Combining all the headers appearing across the train and test set and removing duplicates.
    #A combined list is required because it is possible that some headers only appear in the test set.
    all_asm_list = all_asm_list_train + all_asm_list_test
    all_asm_list = list(set(all_asm_list))
    
    #X_asm_train/test is a 2D list. Each element corresponds to one file...
    #...Each element is a list of the counts of the headers appearing in that file.
    #Headers that do not appear in a file are given a count of 0.
    X_asm_train = prepare_feature_matrix(all_asm_list, asm_each_file_train)
    X_asm_test = prepare_feature_matrix(all_asm_list, asm_each_file_test)

    #Merging the byte count and header count features.
    X_train = [x + y for x, y in zip(X_byte_train, X_asm_train)]
    X_test = [x + y for x, y in zip(X_byte_test, X_asm_test)]

    #Returning a 2D list of features. Each element corresponds to one file...
    #...Each element is a list of the combined byte and header counts.
    return X_train, X_test
    
def get_byte_features(file_list):
    '''This function extracts the byte counts of each byte file.
    It returns a list of bytes that appeared across all the files and...
    ...a list of byte counts for each file.
    '''
    appeared_bytes = {}
    byte_counts = []
    for f in file_list:
        #Create a Response object of a particular byte file.
        r = requests.get("https://storage.googleapis.com/uga-dsp/project1/data/bytes/" + f +".bytes", stream=True)
        #Creating a dictionary with bytes as KEYS and their counts as VALUES.
        byte_counts_file = {}
        #Read the byte file line by line.
        for line in r.iter_lines():
            line = line.decode(errors='ignore')
            if len(line) == 0:
                continue
            #Splitting the line on whitespaces and looping over each byte.
            line = line.split()
            for byte in line[1:]:
                #Checking if the byte has appeared before in any file.
                #If not, adding it to the list of appeared bytes.
                if byte not in appeared_bytes:
                    appeared_bytes[byte] = 1
                #Checking if the byte has appeared before in this particular file.
                #If not, adding it to this file's byte count dictionary.
                #Otherwise incrementing its count.
                if byte not in byte_counts_file:
                    byte_counts_file[byte] = 1
                else:
                    byte_counts_file[byte] += 1
        #Adding byte counts for this file to the byte counts list.
        byte_counts.append(byte_counts_file)
    appeared_bytes_list = list(appeared_bytes.keys())
    return appeared_bytes_list, byte_counts

def get_asm_features(file_list):
    '''This function extracts the header counts of each asm file.
    It returns a list of headers that appeared across all the files and...
    ...a list of header counts for each file.
    '''
    appeared_headers = {}
    header_counts = []
    for f in file_list:
        #Create a Response object of a particular byte file.
        r = requests.get("https://storage.googleapis.com/uga-dsp/project1/data/asm/" + f +".asm", stream=True)
        header_counts_file = {}
        for line in r.iter_lines():
            #Read the asm file line by line.
            line = line.decode(errors='ignore')
            if len(line) == 0:
                continue
            #Splitting the line on ':' and selecting the first element.
            header = line.split(':')[0]
            #Checking if the header has appeared before in any file.
            #If not, adding it to the list of appeared headers.     
            if header not in appeared_headers:
                appeared_headers[header] = 1
            #Checking if the header has appeared before in this particular file.
            #If not, adding it to this file's header count dictionary.
            #Otherwise incrementing its count.
            if header not in header_counts_file:
                header_counts_file[header] = 1
            else:
                header_counts_file[header] += 1
        header_counts.append(header_counts_file)
    #Adding header counts for this file to the header counts list.
    appeared_headers_list = list(appeared_headers.keys())
    return appeared_headers_list, header_counts

def prepare_feature_matrix(feature_list, file_counts_list):
    '''Creates a 2D list of features. Each elements correspnds to features of one file.
    Each element is a list of numerical features.'''
    feature_matrix = []
    #Looping over the list of dictionaries of header/byte : count pairs. 
    for file in file_counts_list:
        file_features = []
        #Looping over all the recorded bytes/headers.
        for feature in feature_list:
            #If a file does not contain this byte/header, its feature value is set to 0.
            #Otherwise it's set to the count.
            if feature not in file:
                file_features.append(0)
            else:
                file_features.append(file[feature])
        #Adding this file'f features to the feature matrix.
        feature_matrix.append(file_features)
    return feature_matrix

def get_labels(y_path):
    '''Creates a list of labels for training.'''
    y_train_file = requests.get('https://storage.googleapis.com/uga-dsp/project1/files/y_small_train.txt', stream=True)
    y_train = []
    for line in y_train_file.iter_lines():
        line = line.decode(errors='ignore')
        #1 is subtracted because spark requires labels start from 0.
        y_train.append(int(line) - 1)
    return y_train

def create_training_matrix(X_train, y_train):
    '''Creates a training matrix to be passed to spark.'''
    data = []
    train_len = len(X_train)    
    for i in range(train_len):
        data.append(LabeledPoint(y_train[i], X_train[i]))
    return data

def predict_and_save(sc, train_data, X_test, num_trees_range=[20, 51], max_depth_range=[10, 31]):
    '''Trains a Random Forest classifier.
    Loops over different values of trees and max depth.'''
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

def main():
    
    #Setting parameters for Spark
    sc = SparkContext.getOrCreate()
    sc.stop()
    conf = SparkConf().setAll([('spark.executor.memory', '7000M'), ('spark.rpc.message.maxSize', '2047'), ('spark.ui.showConsoleProgress', 'true')])
    sc = SparkContext.getOrCreate(conf=conf)
    print(sc.getConf().getAll())
    spark = SparkSession(sc)
    
    #Settin URls for files
    X_train_path = 'https://storage.googleapis.com/uga-dsp/project1/files/X_small_train.txt'
    X_test_path = 'https://storage.googleapis.com/uga-dsp/project1/files/X_small_test.txt'
    y_path = 'https://storage.googleapis.com/uga-dsp/project1/files/y_small_train.txt'
    
    #Creating a list of y labels
    y_train = get_labels(y_path)
    
    #Extracing features from train and test set
    X_train, X_test = get_features(X_train_path, X_test_path)
    
    #Creating a training matrix by combing features and labels
    train_data = create_training_matrix(X_train, y_train)
    
    #Passing training data to the model and outputing prediction
    predict_and_save(sc, train_data, X_test)


if __name__ == "__main__":
    main()
