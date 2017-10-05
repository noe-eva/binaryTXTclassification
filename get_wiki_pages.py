# -*- coding: utf-8 -*-
"""
Date: October 2017
Author: Noëmi Aepli

Script to get the content of a list of wikipedia pages and write it in 
separate files.

Usage: python get_wiki_pages.py INPUT_FOLDER

Input: INPUT_FOLDER containing all ".csv" files extracted from 
"https://petscan.wmflabs.org/", named after their category.

Output: a data folder containing subfolders for each category. 
Subfolders contain a text file for each wikipedia page
listed in the ".csv" file of the respective category (first 1000).

"""

import sys
import wikipedia
import codecs
import os
import csv

def get_pages(in_file, folder):
    ''' extracts the titles of the wikipedia pages
    parameters: in_file = name of csv file
    returns a tuple: list of all titles and the category'''
    category = in_file[:-4] # remove file ending
    title_list = []
    with codecs.open(os.path.join(folder,in_file), "rb", "utf-8") as in_f:
        csv_reader = csv.reader(in_f, delimiter=",")
        next(csv_reader, None) #skip header
        for row in csv_reader:
            title_list.append(row[1])
    return title_list, category
                      

def create_files(page_list, category):
    ''' gets wikipedia page for each title and creates a file with its content
    parameters: list of page titles and the respective category'''
    page_count = 1
    unwanted = set('/()\.')
    for item in page_list:
        if any((char in unwanted) for char in item):
            pass
        else:
            try:            
                out_filename = "data/{}/{}_{}.txt".format(category, str(page_count), item)                
                print("creating file", page_count, out_filename)            
                page = wikipedia.page(item)           
                os.makedirs(os.path.dirname(out_filename), exist_ok=True)
                with codecs.open(out_filename, "w", "utf-8") as f:
                    f.write(page.content)
                page_count += 1
            except (wikipedia.exceptions.PageError, 
                    wikipedia.exceptions.DisambiguationError):
                    pass
        if page_count == 1001:
            break
    return


def main():
    try:
        folder = sys.argv[1]
    except IndexError:
        raise IndexError("Usage: python get_wiki_pages.py INPUT_FOLDER")
    for file in os.listdir(folder):
        print("working on file", file)
        if file.endswith(".csv"):
            title_list, file_category = get_pages(file, folder)
            create_files(title_list, file_category)

if __name__ == '__main__':
    main()