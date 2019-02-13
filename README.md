# Malware Classification

This repository contains the project-1 of Data Science Practicum (CSCI 8360) course at the University of Georgia, Spring 2019. 

This project uses data from the Microsoft Malware Classification Challenge, which consists of nearly half a terabyte of uncompressed data. There are 9 classes of malware, and each instance of malware has one, and only one, category. 

We built a Random Forest classifier which achieves an accuracy of 99.0077%.

Please refer to the [Wiki](https://github.com/dsp-uga/team-dragora-p1/wiki) for more details on our approach.

## Getting Started 

The following instructions will assist you get this project running on your local machine for developing and testing purpose.

### Prerequisites:
- [Apache Spark 2.3.2](https://spark.apache.org/releases/spark-release-2-3-2.html)
- [Python 3.7.2](https://www.python.org/downloads/release/python-372/)
- [Anaconda](https://www.anaconda.com/distribution/)

### Running the tests:

**Run the random forest classifier. The data is automatically pulled from the internet. Byte count or Header count features can be selected (or both).**
```
$ python low_mem_rf.py [Dataset] [Features] [Number of trees] [Maximum depth]
```
OR
```
$ spark-submit low_mem_rf.py [Dataset] [Features] [Number of trees] [Maximum depth]
```
All parameters are optional.

Dataset - ```s``` for small dataset, ```l``` for large dataset. Default: ```l```.

Features - ```11``` for byte and header counts, ```10``` for only byte counts, ```01``` for only header counts. Default: ```11```

Number of trees - Any integer larger than 1. Default: ```40```

Maximum Depth - Any integer between 1 and 30 (inclusive). Default: ```23```

**The prediction will be saved to disk in the same directory and named ```submit<num_trees><max_depth>.txt``` where num_trees is the number of trees and max_depth is the maximum depth.
By default it will be saved as ```submit4023.txt```



**Run the random forest classifier with .dll and opcode count [Optional].**
```
$ spark-submit dllandOpcodefeatutes.py [path-to-opcodeslist.txt]
```
## Authors
(Ordered alphabetically)

- **Anuj Panchmia** 
- **Sumer Singh** 
- **Vishakha Atole**

See the [CONTRIBUTORS.md](https://github.com/dsp-uga/team-dragora-p1/blob/master/CONTRIBUTORS.md) file for details.

## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/dsp-uga/team-dragora-p1/blob/master/LICENSE) file for details
