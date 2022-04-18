# Missing data imputation
# Metrics for calculating differences between complete dataset and dataset with missing info
# Aneja Lab | Yale School of Medicine
# Daniel Yang


# import constants
import pandas as pd
import numpy as np
import math
from sklearn import preprocessing


# Function for calculating RMSE
def rmse(original, missing, imputed, varlist, sublist = None):
    """
    Description:
        This function calculates rmse between an original ("ground truth") dataset 
        and an imputed dataset. values are first 0-1 scaled using sickit MinMaxScaler
    Param:
        original = dataframe of original dataset
        missing = dataframe with spiked-in synthetic missing (np.nan)
        imputed = dataframe with imputed data
        varlist = tuple of lists of variables in each variable type
        sublist = list of variable(s) in category of interest to use to calculate rmse
            default is None in which case returns rmse across all variables
    Return:
        Returns rmse
    """
    # keep only quantitative data type columns in same order
    original = original[varlist[2]]
    missing = missing[varlist[2]]
    imputed = imputed[varlist[2]]
    # normalize data
    normalize = preprocessing.MinMaxScaler().fit(original)
    original_n = pd.DataFrame(normalize.transform(original), columns = original.columns)
    imputed_n = pd.DataFrame(normalize.transform(imputed), columns = imputed.columns)
    # create boolean mask of where data is missing
    bool_mask = (original != missing) & original.notna()
    if sublist == None:
        numerator = (original_n[bool_mask] - imputed_n[bool_mask])**2
        numerator = numerator.sum().sum() #sum once by column then sum overall
        denominator = bool_mask.sum().sum()
    else: # otherwise expect a list of variable names
        numerator = (original_n[sublist][bool_mask] - imputed_n[sublist][bool_mask])**2
        numerator = numerator.sum().sum()
        denominator = bool_mask[sublist].sum().sum()
    return math.sqrt(numerator / denominator)



# Function for calculating PFC
def pfc(original, missing, imputed, varlist, sublist = None):
    """
    Description:
        This function calculates pfc between an original ("ground truth") dataset 
        and an imputed dataset
    Param:
        original = dataframe of original dataset
        missing = dataframe with spiked-in synthetic missing (np.nan)
        imputed = dataframe with imputed data
        varlist = tuple of lists of variables in each variable type
        sublist = list of variable(s) in category of interest to use to calculate pfc
            default is None in which case returns pfc across all variables
    Return:
        Returns pfc
    """
    original = original[varlist[0]+varlist[1]]
    missing = missing[varlist[0]+varlist[1]]
    imputed = imputed[varlist[0]+varlist[1]]
    bool_mask = (original != missing) & original.notna()
    if sublist == None:
        # round since may be comparing floats depending on imputation method
        equal = original[bool_mask].round(1) == imputed[bool_mask].round(1)
        numerator = equal.sum().sum()
        denominator = bool_mask.sum().sum()
    else:
        equal = original[sublist][bool_mask].round(1) == imputed[sublist][bool_mask].round(1)
        numerator = equal.sum().sum()
        denominator = bool_mask[sublist].sum().sum()
    return 1 - numerator / denominator
