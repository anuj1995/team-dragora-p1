from pyspark import SparkConf, SparkContext
import json
import sys
import re
import csv
import requests
from string import punctuation
from pyspark.sql.types import*
from pyspark.sql.functions import*
from pyspark.sql import*
from collections import Counter

conf = (SparkConf()
         .setMaster("yarn")
         .setAppName("asm_features_spark")
         .set('spark.executor.memory', '10G')
         .set('spark.driver.memory', '20G')
         .set('spark.executor.cores', '4')
         )
sc = SparkContext(conf = conf)

def getFileContent(filename):
     #url = "gs://uga-dsp/project1/data/asm/" + filename +".asm"
     url = "https://storage.googleapis.com/uga-dsp/project1/data/asm/"+ filename +".asm"
     output = requests.get(url).text
     wordList = output.lower().split()
     ends_with_dll = list(filter(lambda x: (x.endswith("dll") or x.endswith(".DLL") ), wordList))
     t = dict(Counter(ends_with_dll))
     return (t)

#or x in broadcast_opcode_list

def getfeatureList(wordcountDict,label):
    print(type(wordcountDict))
    print(label)

#Decarling the functions as udf
getFileContent_udf = udf(getFileContent,MapType(StringType(), IntegerType()))
getfeatureList_udf = udf(getfeatureList)


#taking the filename list
url = "https://storage.googleapis.com/uga-dsp/project1/files/X_small_train.txt"
text_file_list = requests.get(url).text
text_file_list = text_file_list.split()

#taking the file labellist
url2 = "https://storage.googleapis.com/uga-dsp/project1/files/y_small_train.txt"
label_file_list = requests.get(url2).text
label_file_list = label_file_list.split()

#taking the opcode list and broadcasting it
url3 = ""
opcode_list = requests.get(url3).text
opcode_list = opcode_list.split()
broadcast_opcode_list = sc.broadcast(opcode_list)

#creating dataframes and joining them
filenames = sc.parallelize(text_file_list,4)
df = filenames.map(lambda x: Row(filename=x)).toDF()
b = spark.createDataFrame([ (l,) for l in label_file_list], ['labelname'])
df = df.withColumn("row_idx", monotonically_increasing_id())
b = b.withColumn("row_idx", monotonically_increasing_id())
final_df = b.join(df, b.row_idx == df.row_idx).drop("row_idx")

# getting the individual word counts
df = final_df.withColumn("counts", getFileContent_udf(final_df['filename']))
exploded_df = df.select("labelname", explode(df['counts']).alias("word","count")).groupBy(df['labelname']).pivot("word").agg(first("count"))

exploded_df.write.csv("/home/dllOpcode.csv")


#dftotal = df.rdd.map(getfeatureList_udf(df['counts'],df['labelname'])).toDF()
