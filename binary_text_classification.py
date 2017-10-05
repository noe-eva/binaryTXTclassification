# -*- coding: utf-8 -*-
"""
Date: October 2017
Author: Noëmi Aepli

Script to train and rank several binary text classifiers.

Usage: python binary_text_classification.py TRAIN_FILES TEST_FILES

Input: TRAIN_FILES/TEST_FILES are the paths to the ".txt" files for training 
and testing respectively.

Output: Classifier ranking (according to F-score)

"""

import sys
import numpy as np
import operator
from collections import defaultdict
from nltk.corpus import stopwords
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import GaussianNB


class BinaryClassifier():
    
    def __init__ (self, train_path, test_path):
        self.train_files = load_files(train_path,'utf-8')
        self.test_files = load_files(test_path, 'utf-8')
        
        self.X_train, self.X_test = self.bow()
        self.X_tfidf = self.tfidf()
        
        self.classifier_scores = defaultdict(int)

    def bow(self):
        '''turn text into numerical feature vectors (bags of words)'''
        print('Creating Bags of Words ...')
        # decode_error=ignore because of .DS_Store files (Mac)
        vectorizer = CountVectorizer(stop_words=stopwords.words('english'), 
                                     encoding='utf-8',  decode_error='ignore')
        X_train = vectorizer.fit_transform(self.train_files.data)
        X_test = vectorizer.transform(self.test_files.data)
        return X_train, X_test


    def tfidf(self):
        '''Calculates Term Frequency (times Inverse Document Frequency; if
        idf=True)'''
        print('Calculating Term Frequencies ...')
        tfidf = TfidfTransformer(use_idf=False)
        X_tfidf = tfidf.fit_transform(self.X_train)
        return X_tfidf


    def best_parameters(self, parameters, classifier):
        '''Perform Grid Search: an exhaustive search over the given parameters
        in order to find the best performing hyperparameters'''        
        gsearch = GridSearchCV(classifier, parameters, verbose=1)
        gsearch.fit(self.X_train, self.train_files.target)
        return gsearch.best_params_


    def classifier_training(self):      
        '''Trains several classifiers and stores results in score dictionary'''        
        
        # Stochastic Gradient Descent Classifier
        print('Training Stochastic Gradient Descent Classifier ...')
        sgd = SGDClassifier(loss='hinge', penalty='l2')
        sgd.fit(self.X_train, self.train_files.target)
        sgd_predict = sgd.predict(self.X_test)
        sgd_score = f1_score(sgd_predict,self.test_files.target)
        self.classifier_scores['Stochastic_Gradient_Descent'] = sgd_score
        
        # Optimised Stochastic Gradient Descent Classifier
        print('Training optimised Stochastic Gradient Descent Classifier ...')
        sgd_parameters = {'loss': ['hinge', 'log', 'modified_huber'], 
                    'penalty': ['l1', 'l2'], 'alpha': 10.0**-np.arange(1,7)}        
        optimised_sgd_parameters = self.best_parameters(sgd_parameters, 
                                                        SGDClassifier())
        sgd_o = SGDClassifier(**optimised_sgd_parameters)
        sgd_o.fit(self.X_train, self.train_files.target)
        sgd_o_predict = sgd_o.predict(self.X_test)
        sgd_o_score = f1_score(sgd_o_predict,self.test_files.target)
        self.classifier_scores['Stochastic_Gradient_Descent_optimised'] = sgd_o_score
        
        # Optimised Stochastic Gradient Descent Classifier with TFIDF       
        print('Training optimised Stochastic Gradient Descent Classifier with TFIDF ...')        
        sgd_o_tfidf = SGDClassifier(**optimised_sgd_parameters)
        sgd_o_tfidf.fit(self.X_tfidf, self.train_files.target)
        sgd_o_tfidf_predict = sgd_o_tfidf.predict(self.X_test)
        sgd_o_tfidf_score = f1_score(sgd_o_tfidf_predict,self.test_files.target)
        self.classifier_scores['Stochastic_Gradient_Descent_tfidf_optimised'] = sgd_o_tfidf_score

        # Gaussian Naive Bayes Classifier     
        print('Training Gaussian Naive Bayes Classifier ...')
        nb = GaussianNB()
        nb.fit(self.X_train.toarray(), self.train_files.target)
        nb_predict = nb.predict(self.X_test.toarray())
        nb_score = f1_score(nb_predict,self.test_files.target)
        self.classifier_scores['Gaussian Naive Bayes'] = nb_score

        # Logistic Regression Classifier    
        print('Training Logistic Regression Classifier ...')
        lr = LogisticRegression()
        lr.fit(self.X_train, self.train_files.target)
        lr_predict = lr.predict(self.X_test)
        lr_score = f1_score(lr_predict,self.test_files.target)
        self.classifier_scores['Logistic_Regression'] = lr_score
 
        # Logistic Regression Classifier with TFIDF            
        print('Training Logistic Regression Classifier with TFIDF ...')
        lr_tfidf = LogisticRegression()
        lr_tfidf.fit(self.X_tfidf, self.train_files.target)
        lr_tfidf_predict = lr_tfidf.predict(self.X_test)
        lr_tfidf_score = f1_score(lr_tfidf_predict,self.test_files.target)
        self.classifier_scores['Logistic_Regression_tfidf'] = lr_tfidf_score
        
        # Optimised Logistic Regression Classifier with TFIDF        
        print('Training optimised Logistic Regression Classifier with TFIDF ...')
        lr_parameters = {'penalty': ['l1', 'l2'], 'C': 
            [0.001, 0.01, 0.1, 1, 10, 100, 1000] }
        optimised_lr_parameters = self.best_parameters(lr_parameters, 
                                                       LogisticRegression())
        lr_tfidf_o = LogisticRegression(**optimised_lr_parameters)
        lr_tfidf_o.fit(self.X_tfidf, self.train_files.target)
        lr_tfidf_o_predict = lr_tfidf_o.predict(self.X_test)
        lr_tfidf_o_score = f1_score(lr_tfidf_o_predict,self.test_files.target)
        self.classifier_scores['Logistic_Regression_tfidf_optimised'] = lr_tfidf_o_score

    def print_results(self):
        '''Prints results of the trained classifiers on the test set, i.e. 
        the dictionnary containing the F1 scores (weighted average of
        precision and recall)'''
        print("\nClassifier ranking:")   
        print("{:38}\t{}". format("Classifier", "F-score"))
        for key, value in sorted(self.classifier_scores.items(), 
                                 key=operator.itemgetter(1), reverse = True):
            print("{:38}\t{}". format(key, value))
    
def main():
    try:
        wiki_binary_classification = BinaryClassifier(sys.argv[1], sys.argv[2])
    except IndexError:
        raise IndexError("Usage: python binary_text_classification.py TRAIN_FILES TEST_FILES")
    wiki_binary_classification.classifier_training()
    wiki_binary_classification.print_results()

if __name__ == '__main__':
    main()