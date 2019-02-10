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


Loading [MathJax]/extensions/Safe.js
Google Cloud DataLab
sec-Copy1
(autosaved)

import requests

from pyspark import SparkContext

from pyspark.sql.session import SparkSession

from pyspark.mllib.regression import LabeledPoint

from pyspark.mllib.tree import GradientBoostedTrees

from pyspark.mllib.tree import RandomForest

import re

import pickle

​

conf = SparkConf().setAll([('spark.executor.memory', '176948M'), ('spark.rpc.message.maxSize', '2047')])

sc.stop()

sc = SparkContext.getOrCreate(conf=conf)

print(sc.getConf().getAll())

spark = SparkSession(sc)

​

​

y_train_file = requests.get('https://storage.googleapis.com/uga-dsp/project1/files/y_train.txt', stream=True)

y_train = []

for line in y_train_file.iter_lines():

    line = line.decode(errors='ignore')

    y_train.append(int(line))

​

y_train = [int(x) - 1 for x in y_train]

​

​

​

with open ('X_train_byte', 'rb') as fp:

    X_train_byte = pickle.load(fp)

​

with open ('X_test_byte', 'rb') as fp:

    X_test_byte = pickle.load(fp)

​

with open ('X_train_asm', 'rb') as fp:

    X_train_asm = pickle.load(fp)

​

with open ('X_test_asm', 'rb') as fp:

    X_test_asm = pickle.load(fp)

​

X_train = [x + y for x, y in zip(X_train_byte, X_train_asm)]

X_test = [x + y for x, y in zip(X_test_byte, X_test_asm)]

​

​

    

data = []

train_len = len(X_train)

for i in range(train_len):

    data.append(LabeledPoint(y_train[i], X_train[i]))

​

model = RandomForest.trainClassifier(sc.parallelize(data), 9, {}, 2000 , seed=42, maxDepth = 30, maxBins = 100)

​

a = []

for i in X_test:

    a.append(int(model.predict(i) + 1))

print(a)

​

​

[('spark.executor.memory', '176948M'), ('spark.dynamicAllocation.minExecutors', '1'), ('spark.driver.maxResultSize', '15360m'), ('spark.eventLog.enabled', 'true'), ('spark.rpc.message.maxSize', '2047'), ('spark.yarn.am.memory', '640m'), ('spark.driver.appUIAddress', 'http://cluster-3-m.us-east1-c.c.p1dsp-230519.internal:4040'), ('spark.driver.host', 'cluster-3-m.us-east1-c.c.p1dsp-230519.internal'), ('spark.executor.instances', '2'), ('spark.driver.port', '45023'), ('spark.yarn.historyServer.address', 'cluster-3-m:18080'), ('spark.serializer.objectStreamReset', '100'), ('spark.org.apache.hadoop.yarn.server.webproxy.amfilter.AmIpFilter.param.PROXY_URI_BASES', 'http://cluster-3-m:8088/proxy/application_1549651215843_0007'), ('spark.app.id', 'application_1549651215843_0007'), ('spark.submit.deployMode', 'client'), ('spark.history.fs.logDirectory', 'hdfs://cluster-3-m/user/spark/eventlog'), ('spark.ui.filters', 'org.apache.hadoop.yarn.server.webproxy.amfilter.AmIpFilter'), ('spark.executor.cores', '8'), ('spark.org.apache.hadoop.yarn.server.webproxy.amfilter.AmIpFilter.param.PROXY_HOSTS', 'cluster-3-m'), ('spark.shuffle.service.enabled', 'true'), ('spark.executorEnv.PYTHONPATH', '/env/python:/usr/lib/spark/python/lib/py4j-0.10.7-src.zip:/usr/lib/spark/python/lib/pyspark.zip<CPS>{{PWD}}/pyspark.zip<CPS>{{PWD}}/py4j-0.10.7-src.zip'), ('spark.scheduler.mode', 'FAIR'), ('spark.hadoop.hive.execution.engine', 'mr'), ('spark.yarn.jars', 'local:/usr/lib/spark/jars/*'), ('spark.scheduler.minRegisteredResourcesRatio', '0.0'), ('spark.executor.id', 'driver'), ('spark.driver.memory', '30720m'), ('spark.app.name', 'pyspark-shell'), ('spark.dynamicAllocation.maxExecutors', '10000'), ('spark.executor.extraJavaOptions', '-Dflogger.backend_factory=com.google.cloud.hadoop.repackaged.gcs.com.google.common.flogger.backend.log4j.Log4jBackendFactory#getInstance'), ('spark.master', 'yarn'), ('spark.executorEnv.PYTHONHASHSEED', '0'), ('spark.rdd.compress', 'True'), ('spark.sql.warehouse.dir', '/root/spark-warehouse'), ('spark.executorEnv.OPENBLAS_NUM_THREADS', '1'), ('spark.yarn.isPython', 'true'), ('spark.eventLog.dir', 'hdfs://cluster-3-m/user/spark/eventlog'), ('spark.sql.parquet.cacheMetadata', 'false'), ('spark.dynamicAllocation.enabled', 'true'), ('spark.ui.proxyBase', '/proxy/application_1549651215843_0006'), ('spark.ui.showConsoleProgress', 'true'), ('spark.sql.cbo.enabled', 'true'), ('spark.driver.extraJavaOptions', '-Dflogger.backend_factory=com.google.cloud.hadoop.repackaged.gcs.com.google.common.flogger.backend.log4j.Log4jBackendFactory#getInstance')]

---------------------------------------------------------------------------
Py4JJavaError                             Traceback (most recent call last)
<ipython-input-1-dd03c487bdc4> in <module>()
     47     data.append(LabeledPoint(y_train[i], X_train[i]))
     48 
---> 49 model = RandomForest.trainClassifier(sc.parallelize(data), 9, {}, 2000 , seed=42, maxDepth = 30, maxBins = 100)
     50 
     51 a = []

/usr/lib/spark/python/lib/pyspark.zip/pyspark/mllib/tree.py in trainClassifier(cls, data, numClasses, categoricalFeaturesInfo, numTrees, featureSubsetStrategy, impurity, maxDepth, maxBins, seed)
    404         return cls._train(data, "classification", numClasses,
    405                           categoricalFeaturesInfo, numTrees, featureSubsetStrategy, impurity,
--> 406                           maxDepth, maxBins, seed)
    407 
    408     @classmethod

/usr/lib/spark/python/lib/pyspark.zip/pyspark/mllib/tree.py in _train(cls, data, algo, numClasses, categoricalFeaturesInfo, numTrees, featureSubsetStrategy, impurity, maxDepth, maxBins, seed)
    303     def _train(cls, data, algo, numClasses, categoricalFeaturesInfo, numTrees,
    304                featureSubsetStrategy, impurity, maxDepth, maxBins, seed):
--> 305         first = data.first()
    306         assert isinstance(first, LabeledPoint), "the data should be RDD of LabeledPoint"
    307         if featureSubsetStrategy not in cls.supportedFeatureSubsetStrategies:

/usr/lib/spark/python/lib/pyspark.zip/pyspark/rdd.py in first(self)
   1374         ValueError: RDD is empty
   1375         """
-> 1376         rs = self.take(1)
   1377         if rs:
   1378             return rs[0]

/usr/lib/spark/python/lib/pyspark.zip/pyspark/rdd.py in take(self, num)
   1356 
   1357             p = range(partsScanned, min(partsScanned + numPartsToTry, totalParts))
-> 1358             res = self.context.runJob(self, takeUpToNumLeft, p)
   1359 
   1360             items += res

/usr/lib/spark/python/lib/pyspark.zip/pyspark/context.py in runJob(self, rdd, partitionFunc, partitions, allowLocal)
   1031         # SparkContext#runJob.
   1032         mappedRDD = rdd.mapPartitions(partitionFunc)
-> 1033         sock_info = self._jvm.PythonRDD.runJob(self._jsc.sc(), mappedRDD._jrdd, partitions)
   1034         return list(_load_from_socket(sock_info, mappedRDD._jrdd_deserializer))
   1035 

/usr/lib/spark/python/lib/py4j-0.10.7-src.zip/py4j/java_gateway.py in __call__(self, *args)
   1255         answer = self.gateway_client.send_command(command)
   1256         return_value = get_return_value(
-> 1257             answer, self.gateway_client, self.target_id, self.name)
   1258 
   1259         for temp_arg in temp_args:

/usr/lib/spark/python/lib/pyspark.zip/pyspark/sql/utils.py in deco(*a, **kw)
     61     def deco(*a, **kw):
     62         try:
---> 63             return f(*a, **kw)
     64         except py4j.protocol.Py4JJavaError as e:
     65             s = e.java_exception.toString()

/usr/lib/spark/python/lib/py4j-0.10.7-src.zip/py4j/protocol.py in get_return_value(answer, gateway_client, target_id, name)
    326                 raise Py4JJavaError(
    327                     "An error occurred while calling {0}{1}{2}.\n".
--> 328                     format(target_id, ".", name), value)
    329             else:
    330                 raise Py4JError(

Py4JJavaError: An error occurred while calling z:org.apache.spark.api.python.PythonRDD.runJob.
: org.apache.spark.SparkException: Job aborted due to stage failure: Task 0 in stage 0.0 failed 4 times, most recent failure: Lost task 0.3 in stage 0.0 (TID 3, cluster-3-w-0.us-east1-c.c.p1dsp-230519.internal, executor 2): org.apache.spark.api.python.PythonException: Traceback (most recent call last):
  File "/usr/lib/spark/python/lib/pyspark.zip/pyspark/worker.py", line 253, in main
    process()
  File "/usr/lib/spark/python/lib/pyspark.zip/pyspark/worker.py", line 248, in process
    serializer.dump_stream(func(split_index, iterator), outfile)
  File "/usr/lib/spark/python/lib/pyspark.zip/pyspark/serializers.py", line 379, in dump_stream
    vs = list(itertools.islice(iterator, batch))
  File "/usr/lib/spark/python/lib/pyspark.zip/pyspark/rdd.py", line 1352, in takeUpToNumLeft
  File "/usr/lib/spark/python/lib/pyspark.zip/pyspark/serializers.py", line 146, in load_stream
    yield self._read_with_length(stream)
  File "/usr/lib/spark/python/lib/pyspark.zip/pyspark/serializers.py", line 171, in _read_with_length
    return self.loads(obj)
  File "/usr/lib/spark/python/lib/pyspark.zip/pyspark/serializers.py", line 566, in loads
    return pickle.loads(obj, encoding=encoding)
  File "/usr/lib/spark/python/lib/pyspark.zip/pyspark/mllib/__init__.py", line 28, in <module>
    import numpy
ImportError: No module named 'numpy'

    at org.apache.spark.api.python.BasePythonRunner$ReaderIterator.handlePythonException(PythonRunner.scala:330)
    at org.apache.spark.api.python.PythonRunner$$anon$1.read(PythonRunner.scala:470)
    at org.apache.spark.api.python.PythonRunner$$anon$1.read(PythonRunner.scala:453)
    at org.apache.spark.api.python.BasePythonRunner$ReaderIterator.hasNext(PythonRunner.scala:284)
    at org.apache.spark.InterruptibleIterator.hasNext(InterruptibleIterator.scala:37)
    at scala.collection.Iterator$class.foreach(Iterator.scala:893)
    at org.apache.spark.InterruptibleIterator.foreach(InterruptibleIterator.scala:28)
    at scala.collection.generic.Growable$class.$plus$plus$eq(Growable.scala:59)
    at scala.collection.mutable.ArrayBuffer.$plus$plus$eq(ArrayBuffer.scala:104)
    at scala.collection.mutable.ArrayBuffer.$plus$plus$eq(ArrayBuffer.scala:48)
    at scala.collection.TraversableOnce$class.to(TraversableOnce.scala:310)
    at org.apache.spark.InterruptibleIterator.to(InterruptibleIterator.scala:28)
    at scala.collection.TraversableOnce$class.toBuffer(TraversableOnce.scala:302)
    at org.apache.spark.InterruptibleIterator.toBuffer(InterruptibleIterator.scala:28)
    at scala.collection.TraversableOnce$class.toArray(TraversableOnce.scala:289)
    at org.apache.spark.InterruptibleIterator.toArray(InterruptibleIterator.scala:28)
    at org.apache.spark.api.python.PythonRDD$$anonfun$3.apply(PythonRDD.scala:152)
    at org.apache.spark.api.python.PythonRDD$$anonfun$3.apply(PythonRDD.scala:152)
    at org.apache.spark.SparkContext$$anonfun$runJob$5.apply(SparkContext.scala:2074)
    at org.apache.spark.SparkContext$$anonfun$runJob$5.apply(SparkContext.scala:2074)
    at org.apache.spark.scheduler.ResultTask.runTask(ResultTask.scala:87)
    at org.apache.spark.scheduler.Task.run(Task.scala:109)
    at org.apache.spark.executor.Executor$TaskRunner.run(Executor.scala:345)
    at java.util.concurrent.ThreadPoolExecutor.runWorker(ThreadPoolExecutor.java:1149)
    at java.util.concurrent.ThreadPoolExecutor$Worker.run(ThreadPoolExecutor.java:624)
    at java.lang.Thread.run(Thread.java:748)

Driver stacktrace:
    at org.apache.spark.scheduler.DAGScheduler.org$apache$spark$scheduler$DAGScheduler$$failJobAndIndependentStages(DAGScheduler.scala:1651)
    at org.apache.spark.scheduler.DAGScheduler$$anonfun$abortStage$1.apply(DAGScheduler.scala:1639)
    at org.apache.spark.scheduler.DAGScheduler$$anonfun$abortStage$1.apply(DAGScheduler.scala:1638)
    at scala.collection.mutable.ResizableArray$class.foreach(ResizableArray.scala:59)
    at scala.collection.mutable.ArrayBuffer.foreach(ArrayBuffer.scala:48)
    at org.apache.spark.scheduler.DAGScheduler.abortStage(DAGScheduler.scala:1638)
    at org.apache.spark.scheduler.DAGScheduler$$anonfun$handleTaskSetFailed$1.apply(DAGScheduler.scala:831)
    at org.apache.spark.scheduler.DAGScheduler$$anonfun$handleTaskSetFailed$1.apply(DAGScheduler.scala:831)
    at scala.Option.foreach(Option.scala:257)
    at org.apache.spark.scheduler.DAGScheduler.handleTaskSetFailed(DAGScheduler.scala:831)
    at org.apache.spark.scheduler.DAGSchedulerEventProcessLoop.doOnReceive(DAGScheduler.scala:1872)
    at org.apache.spark.scheduler.DAGSchedulerEventProcessLoop.onReceive(DAGScheduler.scala:1821)
    at org.apache.spark.scheduler.DAGSchedulerEventProcessLoop.onReceive(DAGScheduler.scala:1810)
    at org.apache.spark.util.EventLoop$$anon$1.run(EventLoop.scala:48)
    at org.apache.spark.scheduler.DAGScheduler.runJob(DAGScheduler.scala:642)
    at org.apache.spark.SparkContext.runJob(SparkContext.scala:2034)
    at org.apache.spark.SparkContext.runJob(SparkContext.scala:2055)
    at org.apache.spark.SparkContext.runJob(SparkContext.scala:2074)
    at org.apache.spark.api.python.PythonRDD$.runJob(PythonRDD.scala:152)
    at org.apache.spark.api.python.PythonRDD.runJob(PythonRDD.scala)
    at sun.reflect.NativeMethodAccessorImpl.invoke0(Native Method)
    at sun.reflect.NativeMethodAccessorImpl.invoke(NativeMethodAccessorImpl.java:62)
    at sun.reflect.DelegatingMethodAccessorImpl.invoke(DelegatingMethodAccessorImpl.java:43)
    at java.lang.reflect.Method.invoke(Method.java:498)
    at py4j.reflection.MethodInvoker.invoke(MethodInvoker.java:244)
    at py4j.reflection.ReflectionEngine.invoke(ReflectionEngine.java:357)
    at py4j.Gateway.invoke(Gateway.java:282)
    at py4j.commands.AbstractCommand.invokeMethod(AbstractCommand.java:132)
    at py4j.commands.CallCommand.execute(CallCommand.java:79)
    at py4j.GatewayConnection.run(GatewayConnection.java:238)
    at java.lang.Thread.run(Thread.java:748)
Caused by: org.apache.spark.api.python.PythonException: Traceback (most recent call last):
  File "/usr/lib/spark/python/lib/pyspark.zip/pyspark/worker.py", line 253, in main
    process()
  File "/usr/lib/spark/python/lib/pyspark.zip/pyspark/worker.py", line 248, in process
    serializer.dump_stream(func(split_index, iterator), outfile)
  File "/usr/lib/spark/python/lib/pyspark.zip/pyspark/serializers.py", line 379, in dump_stream
    vs = list(itertools.islice(iterator, batch))
  File "/usr/lib/spark/python/lib/pyspark.zip/pyspark/rdd.py", line 1352, in takeUpToNumLeft
  File "/usr/lib/spark/python/lib/pyspark.zip/pyspark/serializers.py", line 146, in load_stream
    yield self._read_with_length(stream)
  File "/usr/lib/spark/python/lib/pyspark.zip/pyspark/serializers.py", line 171, in _read_with_length
    return self.loads(obj)
  File "/usr/lib/spark/python/lib/pyspark.zip/pyspark/serializers.py", line 566, in loads
    return pickle.loads(obj, encoding=encoding)
  File "/usr/lib/spark/python/lib/pyspark.zip/pyspark/mllib/__init__.py", line 28, in <module>
    import numpy
ImportError: No module named 'numpy'

    at org.apache.spark.api.python.BasePythonRunner$ReaderIterator.handlePythonException(PythonRunner.scala:330)
    at org.apache.spark.api.python.PythonRunner$$anon$1.read(PythonRunner.scala:470)
    at org.apache.spark.api.python.PythonRunner$$anon$1.read(PythonRunner.scala:453)
    at org.apache.spark.api.python.BasePythonRunner$ReaderIterator.hasNext(PythonRunner.scala:284)
    at org.apache.spark.InterruptibleIterator.hasNext(InterruptibleIterator.scala:37)
    at scala.collection.Iterator$class.foreach(Iterator.scala:893)
    at org.apache.spark.InterruptibleIterator.foreach(InterruptibleIterator.scala:28)
    at scala.collection.generic.Growable$class.$plus$plus$eq(Growable.scala:59)
    at scala.collection.mutable.ArrayBuffer.$plus$plus$eq(ArrayBuffer.scala:104)
    at scala.collection.mutable.ArrayBuffer.$plus$plus$eq(ArrayBuffer.scala:48)
    at scala.collection.TraversableOnce$class.to(TraversableOnce.scala:310)
    at org.apache.spark.InterruptibleIterator.to(InterruptibleIterator.scala:28)
    at scala.collection.TraversableOnce$class.toBuffer(TraversableOnce.scala:302)
    at org.apache.spark.InterruptibleIterator.toBuffer(InterruptibleIterator.scala:28)
    at scala.collection.TraversableOnce$class.toArray(TraversableOnce.scala:289)
    at org.apache.spark.InterruptibleIterator.toArray(InterruptibleIterator.scala:28)
    at org.apache.spark.api.python.PythonRDD$$anonfun$3.apply(PythonRDD.scala:152)
    at org.apache.spark.api.python.PythonRDD$$anonfun$3.apply(PythonRDD.scala:152)
    at org.apache.spark.SparkContext$$anonfun$runJob$5.apply(SparkContext.scala:2074)
    at org.apache.spark.SparkContext$$anonfun$runJob$5.apply(SparkContext.scala:2074)
    at org.apache.spark.scheduler.ResultTask.runTask(ResultTask.scala:87)
    at org.apache.spark.scheduler.Task.run(Task.scala:109)
    at org.apache.spark.executor.Executor$TaskRunner.run(Executor.scala:345)
    at java.util.concurrent.ThreadPoolExecutor.runWorker(ThreadPoolExecutor.java:1149)
    at java.util.concurrent.ThreadPoolExecutor$Worker.run(ThreadPoolExecutor.java:624)
    ... 1 more


!which python

/usr/local/envs/py3env/bin/python

!conda install -n py3env numpy

​

Solving environment: done


==> WARNING: A newer version of conda exists. <==
  current version: 4.5.12
  latest version: 4.6.2

Please update conda by running

    $ conda update -n base -c defaults conda



## Package Plan ##

  environment location: /usr/local/envs/py3env

  added / updated specs: 
    - numpy


The following packages will be downloaded:

    package                    |            build
    ---------------------------|-----------------
    mkl_random-1.0.1           |   py35h4414c95_1         362 KB  defaults
    ca-certificates-2019.1.23  |                0         126 KB  defaults
    numpy-1.15.2               |   py35h1d66e8a_0          47 KB  defaults
    numpy-base-1.15.2          |   py35h81de0dd_0         4.2 MB  defaults
    mkl_fft-1.0.6              |   py35h7dd41cf_0         149 KB  defaults
    ------------------------------------------------------------
                                           Total:         4.9 MB

The following NEW packages will be INSTALLED:

    mkl_fft:         1.0.6-py35h7dd41cf_0  defaults
    mkl_random:      1.0.1-py35h4414c95_1  defaults
    numpy-base:      1.15.2-py35h81de0dd_0 defaults

The following packages will be UPDATED:

    ca-certificates: 2018.03.07-0          defaults --> 2019.1.23-0           defaults
    numpy:           1.14.0-py35ha266831_2 defaults --> 1.15.2-py35h1d66e8a_0 defaults

Proceed ([y]/n)? 

y

​

conf = SparkConf().setAll([('spark.executor.memory', '176948M'), ('spark.rpc.message.maxSize', '2047')])
sc.stop()
sc = SparkContext.getOrCreate(conf=conf)
print(sc.getConf().getAll())
spark = SparkSession(sc)
import requests
from pyspark import SparkContext
from pyspark.sql.session import SparkSession
from pyspark.sql import Row
from pyspark.sql.functions import udf
from pyspark.ml.feature import CountVectorizer
import re
from pyspark.sql.types import ArrayType, StringType
import pickle
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.linalg import Vectors

train_url = 'https://storage.googleapis.com/uga-dsp/project1/files/X_train.txt'
test_url = 'https://storage.googleapis.com/uga-dsp/project1/files/X_test.txt'
y_train_url = 'https://storage.googleapis.com/uga-dsp/project1/files/y_train.txt'
#y_test_url = 'https://storage.googleapis.com/uga-dsp/project1/files/y_test.txt'
y_train_text = requests.get(y_train_url, stream=True)
#y_test_text = requests.get(y_test_url, stream=True)
y_train = []
#y_test = []
y_train_text = y_train_text.text.split('\n')
#y_test_text = y_test_text.text.split('\n')
for i in y_train_text:
    if len(i) > 0:
        y_train.append(int(i) - 1)
#for i in y_test_text:
 #   if len(i) > 0:
  #      y_test.append(int(i) - 1)

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
    

train_file_contents_bytes = []
train_file_contents_asm = []
count = 0 
for f in train_file_names:
    print(count)
    count+=1
    r = requests.get("https://storage.googleapis.com/uga-dsp/project1/data/bytes/" + f +".bytes", stream=True)
    s = []
    for line in r.iter_lines():
        line = line.decode(errors='ignore')
        if len(line) == 0:
            continue
        line = line.split()
        s.extend(line[1:])
    train_file_contents_bytes.append(s)
    s = []
    r = requests.get("https://storage.googleapis.com/uga-dsp/project1/data/asm/" + f +".asm", stream=True)
    for line in r.iter_lines():
        line = line.decode(errors='ignore')
        if len(line) == 0:
            continue
        line = line.split()
        seg = line[0].split(':')[0]
        s.append(seg)
    train_file_contents_asm.append(s)

    
print('done')
count = 0

test_file_contents_bytes = []
test_file_contents_asm = []
for f in test_file_names:
    print(count)
    count+=1
    r = requests.get("https://storage.googleapis.com/uga-dsp/project1/data/bytes/" + f +".bytes", stream=True)
    s = []
    for line in r.iter_lines():
        line = line.decode(errors='ignore')
        if len(line) == 0:
            continue
        line = line.split()
        s.extend(line[1:])
    test_file_contents_bytes.append(s)
    r = requests.get("https://storage.googleapis.com/uga-dsp/project1/data/asm/" + f +".asm", stream=True)
    s = []
    for line in r.iter_lines():
        line = line.decode(errors='ignore')
        if len(line) == 0:
            continue
        line = line.split()
        seg = line[0].split(':')[0]
        s.append(seg)
    test_file_contents_asm.append(s)



print('done')
R = Row('file_name', 'asm_contents', 'byte_contents', 'labels')



train_df = spark.createDataFrame(R(i, p, x, z) for i, p, x, z in zip(train_file_names, train_file_contents_asm, train_file_contents_bytes, y_train))
#test_df = spark.createDataFrame(R(i, Vectors.dense(o), x, z) for i, o, x, z in zip(test_file_names, X_old_test, test_file_contents, y_test))
test_df = spark.createDataFrame(R(i, p, x) for i, p, x in zip(test_file_names, test_file_contents_asm, test_file_contents_bytes))

print('done')


'''

for j in [clean, clean2, clean3, clean4, clean5, clean6]:
    train_df = train_df.withColumn('byte_contents', j('byte_contents'))
    test_df = test_df.withColumn('byte_contents', j('byte_contents'))
    
for j in [clean_asm]:
    train_df = train_df.withColumn('asm_contents', j('asm_contents'))
    test_df = test_df.withColumn('asm_contents', j('asm_contents'))
'''

print('done')

combine_df = train_df.union(test_df)
cv = CountVectorizer(inputCol="byte_contents", outputCol="byte_vectors")
print(combine_df.dtypes)

model = cv.fit(combine_df)

print('done')

train_df_byte = model.transform(train_df)
test_df_byte = model.transform(test_df)

cv = CountVectorizer(inputCol="asm_contents", outputCol="asm_vectors")

model2 = cv.fit(combine_df)

print('done')

train_df_byte_asm = model2.transform(train_df_byte)
test_df_byte_asm = model2.transform(test_df_byte)

print('done')


                    
assembler = VectorAssembler(
    inputCols=["asm_vectors", "byte_vectors"],
    outputCol="features")
train_final_df = assembler.transform(train_df_byte_asm)
test_final_df = assembler.transform(train_df_byte_asm)

print('done')

model_rf = RandomForest.trainClassifier(labelCol="labels", featuresCol="features", numTrees = 60, seed=42, maxDepth = 30, maxBins = 32, maxMemoryInMB=2048)

print('done')

from pyspark.ml.classification import RandomForestClassifier
rf = RandomForestClassifier(labelCol="labels", featuresCol="features", numTrees = 60, seed=42, maxDepth = 30, maxBins = 32, maxMemoryInMB=2048)
model_rf = rf.fit(train_final_df)

print('done')


prediction = model_rf.transform(test_final_df)

print('done')

result = prediction.select("prediction").collect()

print('done')