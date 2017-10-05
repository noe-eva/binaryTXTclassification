# -*- coding: utf-8 -*-
"""
Date: October 2017
Author: Noëmi Aepli

Script to train a binary classifier using an Artificial Neural Net.

Usage: python binary_text_classification_ann.py TRAIN_FILES TEST_FILES

Input: TRAIN_FILES/TEST_FILES are the paths to the ".txt" files for training 
and testing respectively.

Output: Test and train accuracy of the ANN classifier.

"""

import sys
import pickle
import numpy as np
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer
from theano import tensor as T
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import Adam, SGD


def get_data(train_path, test_path):
    train_files = load_files(train_path,'utf-8')
    test_files = load_files(test_path, 'utf-8')
    
    return train_files, test_files
    

def bow(max_features, train_files, test_files):    
    '''turn text into numerical feature vectors (bags of words)'''

    # get nltk stopwords from a pickle file    
    with open('stopwords.pkl', 'rb') as f:
        stopwords = pickle.load(f)

    # decode_error=ignore because of .DS_Store files (Mac)
    vectorizer = CountVectorizer(stop_words = stopwords, encoding='utf-8', 
                                 max_features=max_features, decode_error='ignore')
    X_train = vectorizer.fit_transform(train_files.data)
    X_test = vectorizer.transform(test_files.data)

    return X_train, X_test
    

def build_model(max_features):
    '''build & compile an ann '''
    model = Sequential()
    model.add(Dense(max_features, input_dim = max_features))
    model.add(Dense(20))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    model.compile(loss='mse', optimizer='adam',metrics=['accuracy'])
    return model


def train_model(model, X_train, train_files, X_test, test_files):
    '''train the compiled ann with the data'''
    model.fit(X_train.toarray(), train_files.target, batch_size = 20, nb_epoch = 4, 
              validation_data=(X_test.toarray(), test_files.target))
    test_score, test_accuracy = model.evaluate(X_test.toarray(), test_files.target, 
                                               batch_size = 20)
    train_score, train_accuracy = model.evaluate(X_train.toarray(), train_files.target, 
                                                 batch_size = 20)   
    return test_accuracy, train_accuracy


def main():
    max_features = 10000 # limit dimensions of data matrix used to train the ann
    try:
        train_files, test_files = get_data(sys.argv[1], sys.argv[2])
    except IndexError:
        raise IndexError("Usage: python binary_text_classification_ann.py TRAIN_FILES TEST_FILES")
    X_train, X_test = bow(max_features, train_files, test_files)
    model = build_model(max_features)
    test_accuracy, train_accuracy = train_model(model, X_train, train_files, 
                                                X_test, test_files)
    print("Result: test accuracy: {}, train accuracy: {}".format(test_accuracy, 
          train_accuracy))   

if __name__ == '__main__':
    main()