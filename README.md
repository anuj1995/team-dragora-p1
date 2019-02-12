# Malware Classification

This repository contains the project-1 of Data Science Practicum (CSCI 8360) course at the University of Georgia,Spring 2019. This project aims at designing a large-scale classifier in Apache Spark that maximizes its classification accuracy against a testing dataset. In this project, (_____ classifier)... is implemented on Malware Classification ......(problem statement)

This project uses data from the Microsoft Malware Classification Challenge, which consists of nearly half a terabyte of uncompressed data. There are no fewer than 9 classes of malware, and each instance of malware has one, and only one, of the following family categories:
  1. Ramnit
  2. Lollipop
  3. Kelihos_ver3 
  4. Vundo
  5. Simda
  6. Tracur
  7. Kelihos_ver1
  8. Obfuscator.ACY 
  9. Gatak
  
Documents are in the form of heaxadecimal binaries that include hash, .asm and .bytes file. The files in the files directory are: 
- X_train_small.txt, y_train_small.txt
- X_test_small.txt, y_test_small.txt 
- X_train.txt, y_train.txt
- X_test.txt

Each X* contains a list of hashes, one per line. Each corresponding y* file is a list of integers, one per line, indicating the malware family to which the binary file with the corresponding hash belongs. 

## Getting Started 

The following instructions will assist you get this project running on your local machine for developing and testing purpose.

### Prerequisites
- [Google Cloud Platform](https://cloud.google.com/)
- [Apache Spark](http://spark.apache.org/)

### Installation

### Usage


## Authors
(Ordered alphabetically)

- **Anuj Panchmia** - [anuj1995](https://github.com/anuj1995)
- **Sumer Singh** - [srs96](https://github.com/srs96)
- **Vishakha Atole** - [vishcocity](https://github.com/vishcocity)

See the [CONTRIBUTORS.md](https://github.com/dsp-uga/team-dragora-p1/blob/master/CONTRIBUTORS.md) file for details.

## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/dsp-uga/team-dragora-p1/blob/master/LICENSE) file for details
