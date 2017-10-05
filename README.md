# Binary Text Classification

This repo provides a little example of tackling a typical machine learning problem like binary text classification from A to Z, i.e. from getting the data to an evaluation of the trained classifier.

The data consists of Wikipedia articles belonging to two categories. [PetScan](https://petscan.wmflabs.org/) is a tool that searches an article category according to specified criteria and is used here in order to get lists of articles belonging to a specific category. These lists are used to get the Wikipedia articles and store them in ".txt" files. This data is then split in training and test set in order to train and evaluate binary text classifiers on it.


## Content


[__wiki__](./wiki): contains ".csv" documents (one per category) downloaded from [PetScan](https://petscan.wmflabs.org/).


[__data__](./data): is the output of ```get_wiki_pages.py```, i.e. a folder per category containing 1000 Wikipedia pages of the respective category as ".txt" files.

 
[__data_split__](./data_split): is the output of ```split_data.py```, containing a folder for the train and test set respectively (here: 900 train files and 100 test files per category).

__results.txt__ is a typical output of ```binary_text_classification(_ann).py```.

__stopwords.pkl__ is a pickled version of the english NLTK stopwords (in case NLTK isn't installed).

[__get_wiki_pages.py__](./get_wiki_pages.py): gets the content of a list of Wikipedia pages and writes each page in a separate ".txt" file. It expects a folder with ".csv" files as input (here: wiki) and produces a folder with the respective Wikipedia pages in ".txt" format as output (here: data).

[__split_data.py__](./split_data.py): splits the data in training and test sets according to the specified percentage (here: 10% as test set).

[__binary_text_classification.py__](./binary_text_classification.py): trains several binary text classifiers and ranks them according to the achieved f-score.


[__binary_text_classification_ann.py__](./binary_text_classification_ann.py): trains a binary classifier using an artificial neural net and outputs its training and test accuracy.


## Dependencies

0. Python
1. [Theano](http://deeplearning.net/software/theano/install.html)
2. Keras, ```sudo pip install keras```
3. numpy, ```sudo pip install numpy```
4. NLTK, ```sudo pip install NLTK```


## Getting started

Get the content of a list of wikipedia pages stored as ".csv" files in a folder (here: wiki):

```
python get_wiki_pages.py wiki
```

Split the data in training and test set according to the specified percentage (here: 10%, i.e. 10% of the data will be the test set):

```
python split_data.py data data_split 10
```

To train some classifiers on the data, use the following command:


```
python binary_text_classification.py data_split/train data_split/test
```

To train a neural network classifier on the data, use the following command:


```
python binary_text_classification_ann.py data_split/train data_split/test
```
