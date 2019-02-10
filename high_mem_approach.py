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
import requests
from pyspark import SparkContext


sc = SparkContext.getOrCreate()
sc.stop()
conf = SparkConf().setAll([('spark.executor.memory', '21576M'), ('spark.rpc.message.maxSize', '2047'), ('spark.ui.showConsoleProgress', 'true')])
sc = SparkContext.getOrCreate(conf=conf)
print(sc.getConf().getAll())
spark = SparkSession(sc)
train_url = 'https://storage.googleapis.com/uga-dsp/project1/files/X_small_train.txt'
test_url = 'https://storage.googleapis.com/uga-dsp/project1/files/X_small_test.txt'
y_train_url = 'https://storage.googleapis.com/uga-dsp/project1/files/y_small_train.txt'
y_train_text = requests.get(y_train_url, stream=True)
y_train = []
y_train_text = y_train_text.text.split('\n')

for i in y_train_text:
    if len(i) > 0:
        y_train.append(int(i) - 1)


train_file_list = requests.get(train_url, stream=True)
test_file_list = requests.get(test_url, stream=True)

train_file_names = []
for line in train_file_list.iter_lines():
    line = line.decode(errors='ignore')
    train_file_names.append(line)
    

test_file_names = []
for line in test_file_list.iter_lines():
    line = line.decode(errors='ignore')
    test_file_names.append(line)


'''
def get_size_features(f):
    response_asm = requests.head("https://storage.googleapis.com/uga-dsp/project1/data/asm/" + f +".asm")
    response_byte = requests.head("https://storage.googleapis.com/uga-dsp/project1/data/bytes/" + f +".bytes")
    size_asm = int(response_asm.headers['Content-Length'])
    size_byte = int(response_byte.headers['Content-Length'])
    size_ratio = size_asm / size_byte   
    return [size_asm, size_byte, size_ratio]
'''

asm_train_content_generator =  (requests.get("https://storage.googleapis.com/uga-dsp/project1/data/asm/" + f +".asm").text for f in train_file_names)
asm_test_content_generator = (requests.get("https://storage.googleapis.com/uga-dsp/project1/data/asm/" + f +".asm").text for f in test_file_names)
byte_train_content_generator = (requests.get("https://storage.googleapis.com/uga-dsp/project1/data/bytes/" + f +".bytes").text for f in train_file_names)
byte_test_content_generator = (requests.get("https://storage.googleapis.com/uga-dsp/project1/data/bytes/" + f +".bytes").text for f in test_file_names)
#file_size_train_content_generator = (get_size_features(f) for f in train_file_names)
#file_size_test_content_generator = (get_size_features(f) for f in test_file_names)



print('done')

R = Row('file_name', 'asm_features', 'byte_feature', 'labels')
train_df = spark.createDataFrame(R(i, s, b, l) for i, s, b, l in zip(train_file_names, asm_train_content_generator, byte_train_content_generator, y_train))
R = Row('file_name', 'asm_features', 'byte_feature')
test_df = spark.createDataFrame(R(i, s, b) for i, s, b in zip(test_file_names, asm_test_content_generator, byte_test_content_generator))


clean = udf(lambda r: " ".join(r.splitlines()))
clean2 = udf(lambda r: re.sub('[A-Z0-9]{8} ', '', r))
clean3 = udf(lambda r: re.sub(' [A-Z0-9]{8}', '', r))
clean4 = udf(lambda r: re.sub(r'\?[A-Z0-9] |[A-Z0-9]\? |\?\? ', '', r))
clean5 = udf(lambda r: re.sub(r' \?[A-Z0-9]| [A-Z0-9]\?| \?\?', '', r))
clean6 = udf(lambda r: r.split(), ArrayType(StringType()))

clean_asm = udf(lambda r: re.findall(r'^([\w\.]+)', r , re.MULTILINE), ArrayType(StringType()))

for j in [clean, clean2, clean3, clean4, clean5, clean6]:
    train_df = train_df.withColumn('byte_feature', j('byte_feature'))
    test_df = test_df.withColumn('byte_feature', j('byte_feature'))

train_df = train_df.withColumn('asm_features', j('asm_features'))
test_df = test_df.withColumn('asm_features', j('asm_features'))

print('done')

combine_df = train_df.union(test_df)
cv = CountVectorizer(inputCol="byte_feature", outputCol="byte_vectors")

model = cv.fit(combine_df)

train_df= model.transform(train_df)
test_df = model.transform(test_df)

cv = CountVectorizer(inputCol="asm_feature", outputCol="asm_vectors")

model = cv.fit(combine_df)

train_df= model.transform(train_df)
test_df = model.transform(test_df)

assembler = VectorAssembler(
    inputCols=["file_size_features"],
    outputCol="file_size_vectors")
train_final = assembler.transform(train_df)
test_final = assembler.transform(test_df)

                    
assembler = VectorAssembler(
    inputCols=["byte_vectors", "asm_vectors", "file_size_vectors"],
    outputCol="features")
train_final_df = assembler.transform(train_df)
test_final_df = assembler.transform(test_df)

print('done')

from pyspark.ml.classification import RandomForestClassifier
rf = RandomForestClassifier(labelCol="labels", featuresCol="features", numTrees = 50, seed=42, maxDepth = 30, maxBins = 32, maxMemoryInMB=2048)
model_rf = rf.fit(train_final_df)

print('done')


prediction = model_rf.transform(test_final_df)

print('done')

result = prediction.select("prediction").collect()

print('done')
