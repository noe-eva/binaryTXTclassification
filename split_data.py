# -*- coding: utf-8 -*-
"""
Date: October 2017
Author: Noëmi Aepli

Script to split files in training and test set.

Usage: python split_data.py INPUT_FOLDER OUTPUT_FOLDER PERCENTAGE

Input:
- INPUT_FOLDER: path to a folder containing folders with files to split
- OUTPUT_FOLDER: name for output folder to be created
- PERCENTAGE: integer between 0 & 100 specifying the percentage of test files

Output: OUTPUT_FOLDER containing train & test folders with files that were 
randomly split according to the specified percentage of test files

"""


import os
import sys
import random
import shutil
import glob


def main():
    
    try:
        superfolder = sys.argv[1]
        new_superfolder = sys.argv[2]
        percentage = int(sys.argv[3])
    except IndexError:
        raise IndexError("Usage: python split_data.py INPUT_FOLDER OUTPUT_FOLDER PERCENTAGE")

    
    try:
        os.makedirs(new_superfolder, exist_ok=True)
    except OSError:
        raise IOError("Can't create directory!")
        
    
    folders = os.listdir(superfolder)
        
    for folder in folders:
       
        new_train_folder = os.path.join(new_superfolder, "train", folder)
        new_test_folder = os.path.join(new_superfolder, "test", folder)
        
        try:
            os.makedirs(new_train_folder, exist_ok=True)
            os.makedirs(new_test_folder, exist_ok=True)
        except OSError:
            raise IOError("Can't create directory!")
    
        path = os.path.join(superfolder, folder)
        
        filenames = glob.glob("{}/*.txt".format(path))
        assert len(filenames) > 0, "No .txt files found in directory."
                    
        # specify the number of files for the test set
        n = len(filenames)*percentage/100
        # choose n random files
        choices = random.sample(filenames, int(n))
        
        # move randomly chosen files to new test folder
        for random_file in choices:
            shutil.move(random_file, new_test_folder)
       
        # move the rest of the files of the same folder to the new train folder 
        filenames = os.listdir(path)
        for file in filenames:
            file_path = os.path.join(path, file) 
            shutil.move(file_path, new_train_folder)

if __name__ == '__main__':
    main()