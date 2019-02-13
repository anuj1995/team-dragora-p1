#importing libraries
from pyspark import SparkConf, SparkContext
from operator import *
import string

#processing the opcode list and broadcasting it
listOfOps = sc.wholeTextFiles("gs://proj-1/opcodelist.txt")
listOfOpsFinal = listOfOps.flatMap(lambda x:x[1].strip().lower().splitlines())
opsBroadcast = sc.broadcast(listOfOpsFinal.collect())
listOfOpsFinal.collect()

#declaring lists for processing
tempa = []
final=[]

#processing the training dataset
listOfFile = sc.wholeTextFiles("gs://uga-dsp/project1/files/X_small_train.txt")
a = listOfFile.flatMap(lambda x: x[1].strip().splitlines())
b = a.map(lambda x: x + ".asm")
listOfFiles=b.collect()
for i in listOfFiles:
    file = sc.wholeTextFiles("gs://uga-dsp/project1/data/asm/"+i)
    fileManipulation = file.flatMap(lambda x:x[1].lower().split())
    fileManipulation1 = fileManipulation.filter(lambda x :x in opsBroadcast.value).collect()
    tempa.append(i+":"+str(fileManipulation1))
for k in tempa:
    listContents =[]
    result = []
    split= k.split(":")
    fileName = split[0]
    listContent = split[1].split(",")
    for i in listContent:
        listContents.append(i.strip().strip(string.punctuation))
    for j in listOfOpsFinal.collect():
        counts = listContents.count(j)
        if int(counts) != 0 :
            res = j+":"+str(counts)
            result.append(res)
    finalRes = fileName+":"+str(result)
    final.append(finalRes)



