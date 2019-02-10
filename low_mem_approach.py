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
    file_names = []
    for line in file_list.iter_lines():
        line = line.decode(errors='ignore')
        file_names.append(line)
    return file_names

def get_features(train_path, test_path):
    train_file = requests.get(train_path, stream=True)
    test_file = requests.get(test_path, stream=True)

    train_file_names = get_file_name(train_file)
    test_file_name = get_file_name(test_file)

    
    all_byte_list_train, byte_each_file_train = get_byte_features(train_file_names)
    all_byte_list_test, byte_each_file_test = get_byte_features(test_file_name)

    all_byte_list = [x + y for x, y in zip(all_byte_list_train, all_byte_list_test)]
    all_byte_list = list(set(combine_segments))
    
    X_byte_train = prepare_feature_matrix(all_byte_list, byte_each_file_train)
    X_byte_test = prepare_feature_matrix(all_byte_list, byte_each_file_test)

    all_asm_list_train, asm_each_file_train = get_asm_features(train_file_names)
    all_asm_list_test, asm_each_file_test = get_asm_features(test_file_name)

    all_asm_list = [x + y for x, y in zip(all_asm_list_train, all_asm_list_test)]
    all_asm_list = list(set(combine_segments))
    
    X_asm_train = prepare_feature_matrix(all_asm_list, asm_each_file_train)
    X_asm_test = prepare_feature_matrix(all_asm_list, asm_each_file_test)

    X_train = [x + y for x, y in zip(X_byte_train, X_asm_train)]
    X_test = [x + y for x, y in zip(X_byte_test, X_asm_test)]

    return X_train, X_test
    
def get_byte_features(file_names):
    all_segments = {}
    segments_each = []
    for f in file_names:
        r = requests.get("https://storage.googleapis.com/uga-dsp/project1/data/bytes/" + f +".bytes", stream=True)
        segments = {}
        for line in r.iter_lines():
            line = line.decode(errors='ignore')
            if len(line) == 0:
                continue
            line = line.split()
            line2 = line[1:]
            
            for i in seg[1:]:
                if seg[0] not in all_segments:
                    all_segments[seg[0]] = 1
                if seg[0] not in segments:
                    segments[seg[0]] = 1
                else:
                    segments[seg[0]] += 1
        segments_each.append(segments)
    all_segments_list = list(all_segments.keys())
    return all_segments_list, segments_each

def get_asm_features(file_list):
    all_segments = {}
    segments_each = []
    for f in file_names:
        print(count)
        r = requests.get("https://storage.googleapis.com/uga-dsp/project1/data/asm/" + f +".asm", stream=True)
        segments = {}
        for line in r.iter_lines():
            line = line.decode(errors='ignore')
            if len(line) == 0:
                continue
            line = line.split()
            seg = line[0].split(':')
            if seg[0] not in all_segments:
                all_segments[seg[0]] = 1
            if seg[0] not in segments:
                segments[seg[0]] = 1
            else:
                segments[seg[0]] += 1
        segments_each.append(segments)
    all_segments_list = list(all_segments.keys())
    return all_segments_list, segments_each

def segment_matrix(combine_segments_list, segments_each):
    segment_feature_matrix = []
    for file in segments_each:
        segment_feature_vector = []
        for seg in combine_segments_list:
            if seg not in file:
                segment_feature_vector.append(0)
            else:
                segment_feature_vector.append(file[seg])
        segment_feature_matrix.append(segment_feature_vector)
   
    return segment_feature_matrix

def_main():
    sc = SparkContext.getOrCreate()
    sc.stop()
    conf = SparkConf().setAll([('spark.executor.memory', '7000M'), ('spark.rpc.message.maxSize', '2047'), ('spark.ui.showConsoleProgress', 'true')])
    sc = SparkContext.getOrCreate(conf=conf)
    print(sc.getConf().getAll())
    spark = SparkSession(sc)

    y_train_file = requests.get('https://storage.googleapis.com/uga-dsp/project1/files/y_train.txt', stream=True)
    y_train = []
    for line in y_train_file.iter_lines():
        line = line.decode(errors='ignore')
        y_train.append(int(line))

    y_train = [int(x) - 1 for x in y_train]

    train_path = 'https://storage.googleapis.com/uga-dsp/project1/files/X_train.txt'
    test_path = 'https://storage.googleapis.com/uga-dsp/project1/files/X_test.txt'

    X_train, X_test = get_features(train_path, test_path)
        
    data = []
    train_len = len(X_train)
    for i in range(train_len):
        data.append(LabeledPoint(y_train[i], X_train[i]))

    
    for trees in range(1, 50):
        for depth in range(1, 30):
            model = RandomForest.trainClassifier(sc.parallelize(data), 9, {}, trees, maxDepth = depth)
            a = []
            for i in X_test:
                 a.append(int(model.predict(i) + 1))
            trees_s = str(trees)
            depth_s = str(depth)
            string = 'submit2' + trees_s + depth_s + '.txt'
            b = pd.DataFrame(a)
            b.to_csv(string, header = False, index = False)

if __name__ == "__main__":
    main()


