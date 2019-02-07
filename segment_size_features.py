sc = SparkContext.getOrCreate()
import requests
from pyspark import SparkContext
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.tree import GradientBoostedTrees
from pyspark.mllib.tree import RandomForest


def get_features(train_path, test_path):

    X_train_segment_features, X_test_segment_features  = segment_features(train_path, test_path)
    X_train_size_features, X_test_size_features = size_features(train_path, test_path)
    X_train = X_train_segment_features[:]
    X_test = X_test_segment_features[:]
    l_train = len(X_train)
    l_test = len(X_test)
    for i in range(l_train):
        X_train[i].extend(X_train_size_features[i])
    for i in range(l_test):
        X_test[i].extend(X_test_size_features[i])
    return X_train, X_test

def size_features(train_path, test_path):
    train_file_list = requests.get(train_path, stream=True)
    test_file_list = requests.get(test_path, stream=True)

    size_train_matrix = get_size_features(train_file_list)
    size_test_matrix = get_size_features(test_file_list)
    return size_train_matrix, size_test_matrix

def get_size_features(file_list):

    file_names = []
    for line in file_list.iter_lines():
        line = line.decode(errors='ignore')
        file_names.append(line)

    size_matrix = []

    for f in file_names:
        response_asm = requests.head("https://storage.googleapis.com/uga-dsp/project1/data/asm/" + f +".asm")
        response_byte = requests.head("https://storage.googleapis.com/uga-dsp/project1/data/bytes/" + f +".bytes")
        size_asm = int(response_asm.headers['Content-Length'])
        size_byte = int(response_byte.headers['Content-Length'])
        size_ratio = size_asm / size_byte
        size_matrix.append([size_asm, size_byte, size_ratio])
    return size_matrix



def segment_features(train_path, test_path):
    train_file_list = requests.get(train_path, stream=True)
    test_file_list = requests.get(test_path, stream=True)
    
    all_segments_train, segments_train = get_segment_features(train_file_list)
    all_segments_test, segments_test = get_segment_features(test_file_list)
    combine_segments = []
    combine_segments.extend(all_segments_train)
    combine_segments.extend(all_segments_test)
    combine_segments = list(set(combine_segments))
    
    X_train = segment_matrix(combine_segments, segments_train)
    X_test = segment_matrix(combine_segments, segments_test)
    
    return X_train, X_test
    
def get_segment_features(file_list):
    file_names = []
    for line in file_list.iter_lines():
        line = line.decode(errors='ignore')
        file_names.append(line)
    all_segments = {}
    segments_each = []
    count = 0 
    for f in file_names:
        print(count)
        count += 1
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

train_url = 'https://storage.googleapis.com/uga-dsp/project1/files/X_train.txt'
test_url = 'https://storage.googleapis.com/uga-dsp/project1/files/X_test.txt'
X_train_features, X_test_features = get_features(train_url, test_url)