# Missing data imputation
# Functions for processing data file
# Aneja Lab | Yale School of Medicine
# Daniel Yang, Miles Hui


#import constants
import pandas as pd
import numpy as np
from sklearn import preprocessing


# Funciton for encoding object data types
def encode_obj(dataframe):
    """
    Description:
        This function encodes a dataframe holding objects and encodes each category 
        with a numeric and converts to float
    Param:
        dataframe = pandas dataframe
    Return:
        Returns a dataframe where each object column is encoded to float
    """
    le = preprocessing.LabelEncoder()
    df_obj = dataframe.select_dtypes(include='object')
    for col in df_obj.columns:
        dataframe.loc[:, col] = le.fit_transform(df_obj[col])
    dataframe = dataframe.astype(float)
    return dataframe


# Function for separating out variable types
def variable_type(dataframe, vars_df):
    """
    Description:
        This function obtains the names of nominal vs ordinal vs quantitative (numeric)
        variables
    Param:
        dataframe = pandas dataframe containing all data
        vars_df = user input csv with variable type
            1 = nominal
            2 or 3 = ordinal
            4 or 5 = quantitative
    Return:
        Returns tuple of lists corresponding to nominal, ordinal, quantitative 
        (numeric) variables        
    """
    vars_nominal = []
    vars_ordinal = []
    vars_quant = []
    for x in dataframe.columns.values:
        if x in vars_df.loc[vars_df['type']==1]['variable'].unique():
            vars_nominal = vars_nominal + [x]
        elif x in vars_df.loc[(vars_df['type']==2) | (vars_df['type']==3)]['variable'].unique():
            vars_ordinal = vars_ordinal + [x]
        else:
            vars_quant = vars_quant + [x]
    return vars_nominal, vars_ordinal, vars_quant


# Function for one hot encoding categorical variables
def one_hot_encode(dataframe, varlist):
    """
    Description:
        One hot encode categorical variables
    Param:
        dataframe = pandas dataframe (can be mixed datatypes)
        varlist = tuple of nominal, ordinal, and quantitative variables
    Return:
        Returns tuple of (encoded dataframe, list of list encoded columns)
    """
    encoded_dataframe = dataframe[varlist[2]] # initialize df_ohe with just the quant variables 
    encoded_colnames = []
    for col in varlist[0]+varlist[1]:
        temp_df = pd.get_dummies(dataframe[col], columns=col, prefix=col, dummy_na=True)
        temp_df.loc[temp_df[col+'_nan']==1] = np.nan
        temp_df = temp_df.drop([col+'_nan'], axis=1)
        encoded_colnames.append(temp_df.columns.tolist())
        encoded_dataframe = pd.concat([encoded_dataframe, temp_df], axis=1)
    return encoded_dataframe, encoded_colnames


# Function for "decoding" one hot encoded categorical variables to its original representation
def ohe_decode(dataframe, varlist):
    """
    Description:
        Decodes one hot encoded dataframe back to original, taking the max of each
        category (df.idxmax) as the imputed value for categorical variables
    Param:
        dataframe = pandas dataframe with categorical columns encoded
        varlist = tuple of nominal, ordinal, and quantitative variables
    Return:
        Returns decoded dataframe
    """
    decoded_dataframe = dataframe[varlist[2]] # initialize with just the quant columns with imputed values
    for col in varlist[0]+varlist[1]:
        temp_df = dataframe.filter(regex='^'+col+'_[0-9]', axis=1)
        temp_df = temp_df.idxmax(axis=1).str.rsplit('_', 1, expand=True)
        temp_df.rename(columns={1:col}, inplace=True)
        temp_df[col] = temp_df[col].astype(float)
        decoded_dataframe = pd.concat([decoded_dataframe, temp_df[col]], axis = 1)
    return decoded_dataframe

# Function to change dataset to independent variables and dependent variables
def split_df(df_train, df_test):
    """
    Description:
        Function to change dataset to independent variables and dependent variables
    Param:
        df_train = training set
        df_test = testing dataset
    Return:
        Returns independent variables: X_train and X_test, and dependent variables: y_train, y_test
    """

    X_train = df_train.copy()
    y_train = df_train[['Event', 'Time']].copy()
    X_test = df_test.copy()
    y_test = df_test[['Event', 'Time']].copy()

    X_train = X_train.drop(columns=['Time', 'Event'])
    y_train['Event'] = y_train['Event'].astype('bool')
    y_train['Time'] = y_train['Time'].astype('float64')
    X_test = X_test.drop(columns=['Time', 'Event'])
    y_test['Event'] = y_test['Event'].astype('bool')
    y_test['Time'] = y_test['Time'].astype('float64')

    return X_train, X_test, y_train, y_test