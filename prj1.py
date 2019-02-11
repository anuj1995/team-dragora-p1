#importing libraries
from pyspark import SparkConf, SparkContext
from operator import *

listOfOps = sc.wholeTextFiles("gs://proj-1/opcodelist.txt")
listOfOpsFinal = listOfOps.flatMap(lambda x:x[1].strip().lower().splitlines())
opsBroadcast = sc.broadcast(listOfOpsFinal.collect())
listOfOpsFinal.collect()


tempa = []
final=[]

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
    result =[]
    split= k.split(":")
    fileName = k[0]
    listContents = list(k[1])
    listContents.remove(",")
    listContents.remove("]")
    listContents.remove("[")
    for j in listOfOpsFinal.collect():
        counts = listContents.count(j)
        if counts != 0 :
            res = j+":"+str(counts)
            result.append(res)
    finalRes = fileName+":"+str(result)
    final.append(finalRes)



import pickle
with open('outfile','wb') as fp:
    pickle.dump(result,fp)


from pyspark.sql.types import StringType
spark.createDataFrame(mylist, StringType()).show()
