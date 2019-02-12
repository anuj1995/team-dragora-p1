from pyspark import SparkConf, SparkContext
import json
import sys
import re
import csv
import string
import requests
from string import punctuation
from pyspark.sql.types import*
from pyspark.sql.functions import*
from pyspark.sql import*
from collections import Counter
from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.feature import StringIndexer, VectorIndexer, VectorAssembler
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import copy

conf = (SparkConf()
         .setMaster("yarn")
         .setAppName("asm_features_spark")
         .set('spark.executor.memory', '10G')
         .set('spark.driver.memory', '20G')
         .set('spark.executor.cores', '4')
         )
sc = SparkContext(conf = conf)

def getWordCount(filename):
    #url = "gs://uga-dsp/project1/data/asm/" + filename +".asm"
    url = "https://storage.googleapis.com/uga-dsp/project1/data/asm/"+ filename +".asm"
    #url = "https://storage.googleapis.com/uga-dsp/project1/data/asm/01IsoiSMh5gxyDYTl4CB.asm"
    output = requests.get(url).text
    wordList = output.lower().split()
    stripedWords = map(lambda x: (x.strip(punctuation)), wordList)
    features = list(filter(lambda x: (x.endswith(".dll") or x.endswith(".DLL") or x in broadcast_opcode_list.value), stripedWords))
    t = dict(Counter(features))
    return (t)

def getTestWordCount(filename):
    #url = "gs://uga-dsp/project1/data/asm/" + filename +".asm"
    url = "https://storage.googleapis.com/uga-dsp/project1/data/asm/"+ filename +".asm"
    #url = "https://storage.googleapis.com/uga-dsp/project1/data/asm/01IsoiSMh5gxyDYTl4CB.asm"
    output = requests.get(url).text
    wordList = output.lower().split()
    stripedWords = map(lambda x: (x.strip(punctuation)), wordList)
    features = list(filter((lambda x : x in test_feature_list.value), stripedWords))
    t = dict(Counter(features))
    return (t)


#or x in broadcast_opcode_list
#hdfs://cluster-64de-m/home/dllOpcode.csv
#Decarling the functions as udf
getWordCount_udf = udf(getWordCount,MapType(StringType(), IntegerType()))
getTestWordCount_udf = udf(getTestWordCount,MapType(StringType(), IntegerType()))

#taking the filename list
url = "https://storage.googleapis.com/uga-dsp/project1/files/X_small_train.txt"
train_text_file_list = requests.get(url).text
train_text_file_list = train_text_file_list.split()

#taking the file labellist
url2 = "https://storage.googleapis.com/uga-dsp/project1/files/y_small_train.txt"
train_label_file_list = requests.get(url2).text
train_label_file_list = train_label_file_list.split()

#taking the opcode list and broadcasting it
url3 = "hdfs://cluster-b24c-m/user/op.txt"
opcode_list = sc.textFile(url3)
opcode_list = opcode_list.flatMap(lambda line: line.lower().split()).collect()
broadcast_opcode_list = sc.broadcast(opcode_list)

url4 = "https://storage.googleapis.com/uga-dsp/project1/files/y_small_test.txt"
test_label_file_list = requests.get(url4).text
test_label_file_list = test_label_file_list.split()

url5 = "https://storage.googleapis.com/uga-dsp/project1/files/X_small_test.txt"
test_text_file_list = requests.get(url5).text
test_text_file_list = test_text_file_list.split()

# Trainig set features ###############################################################################

#creating dataframes and joining them
filenames = spark.createDataFrame([ (t,) for t in train_text_file_list], ['filename'])
filenames = filenames.rdd.zipWithIndex().map(lambda x: (str(x[0].filename),x[1])).toDF()
filenames = filenames.selectExpr("_1 as filename", "_2 as id")
labelnames = spark.createDataFrame([ (l,) for l in train_label_file_list], ['labelname'])
labelnames = labelnames.rdd.zipWithIndex().map(lambda x: (int(x[0].labelname),x[1])).toDF()
labelnames = labelnames.selectExpr("_1 as labelname", "_2 as id")
final_df = labelnames.join(filenames, labelnames.id == filenames.id).drop("id")

# getting the individual word counts
interdf = final_df.withColumn("counts", getWordCount_udf(final_df['filename']))
exploded_df = interdf.select("labelname","filename", explode(interdf['counts']).alias("word","count")).groupBy(interdf['filename']).pivot("word").agg(first("count"))
train_exploded_df = exploded_df.fillna(0)
#exploded_df.coalesce(1).write.format("com.databricks.spark.csv").option("header", "true").save("/home/dllOpcode4.csv")

train_final_df = final_df.join(train_exploded_df, final_df.filename == train_exploded_df.filename).drop("filename")
#dftotal = df.rdd.map(getfeatureList_udf(df['counts'],df['labelname'])).toDF()
train_columns = train_final_df.columns[1:]

# Testing set features ##############################################################################
test_feature_list = sc.broadcast(train_columns)
filenames = spark.createDataFrame([ (t,) for t in test_text_file_list], ['filename'])
filenames = filenames.rdd.zipWithIndex().map(lambda x: (str(x[0].filename),x[1])).toDF()
filenames = filenames.selectExpr("_1 as filename", "_2 as id")
labelnames = spark.createDataFrame([ (l,) for l in test_label_file_list], ['labelname'])
labelnames = labelnames.rdd.zipWithIndex().map(lambda x: (int(x[0].labelname),x[1])).toDF()
labelnames = labelnames.selectExpr("_1 as labelname", "_2 as id")
final_df = labelnames.join(filenames, labelnames.id == filenames.id).drop("id")

interdf = final_df.withColumn("counts", getTestWordCount_udf(final_df['filename']))
exploded_df = interdf.select("labelname","filename", explode(interdf['counts']).alias("word","count")).groupBy(interdf['filename']).pivot("word").agg(first("count"))
test_exploded_df = exploded_df.fillna(0)
test_final_df = final_df.join(test_exploded_df, final_df.filename == test_exploded_df.filename).drop("filename")
test_columns = test_final_df.columns[1:]
for x in test_columns:
    train_columns.remove(x)
cols = ["*"] + [lit(None).cast("int").alias(feature) for feature in train_columns]
temp = test_final_df.select(cols)
test_final_df = temp.fillna(0)


#removing the "." from the column names
oldColumns = train_final_df.schema.names
newColumns =list()
for i in oldColumns:
    newColumns.append(i.replace(".",""))

train_final_df = train_final_df.toDF(*newColumns)
test_final_df = test_final_df.toDF(*newColumns)

# randon forrest classifier ##############################################################################
assembler = VectorAssembler(inputCols=train_final_df.columns[1:], outputCol="features")
dt = DecisionTreeClassifier(labelCol="labelname", featuresCol="features")
pipeline = Pipeline(stages=[assembler, dt])
model = pipeline.fit(train_final_df)
predictions = model.transform(test_final_df)
