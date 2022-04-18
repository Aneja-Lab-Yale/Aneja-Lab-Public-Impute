# Missing data imputation
# File for defining constants
# Aneja Lab | Yale School of Medicine
# Daniel Yang, Miles Hui

import os
# working folder
seed = 10
DATA_FOLDER = './data/'

# data name will be the last dta file in ./data/ folder
for file in os.listdir('./data/'):
    if file.endswith(".dta"):
        DATA_NAME = file

RESULT_FOLDER = './result/'
# csv file where each variable type is labeled
VARIABLES_TYPE = './data/variables.csv'
