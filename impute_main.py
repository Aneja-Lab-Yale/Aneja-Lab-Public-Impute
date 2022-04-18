# Missing data imputation
# Main file for running imputation methods
# Aneja Lab | Yale School of Medicine
# Daniel Yang, Miles Hui

import os
import numpy as np
import pandas as pd
import time

import tensorflow as tf
import random

from fancyimpute import SoftImpute
from fancyimpute import IterativeSVD
from missingpy import MissForest
from MIDASpy import Midas


from sklearn.experimental import enable_iterative_imputer # be sure to keep this line
from sklearn.impute import IterativeImputer
from sklearn.impute import KNNImputer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler


# import user written modules and constants
import process_data
from constants import DATA_FOLDER, RESULT_FOLDER, DATA_NAME
from constants import VARIABLES_TYPE
from constants import seed


random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

# load data
df = pd.read_stata(DATA_FOLDER + DATA_NAME)
df = df.astype(float) # convert all to float
# x is a tuple where x[0]=categorical; x[1]=ordinal; x[2]=continuous
# x[0]+x[1] is treated as categorical in paper
variables = pd.read_csv(VARIABLES_TYPE, header=0)
x = process_data.variable_type(df, variables)


# Function for spiking in missingness
bool_stage4 = (df['ANALYTIC_STAGE_GROUP'] == 4)
def spike_missing(dataframe, percent=0.1, mechanism = 'MCAR', indicator = bool_stage4):
    """
    Param:
        dataframe = dataframe to spike in additional missing
        percent = percentage of rows in each column to make additionally missing
        mechanism = missingness mechanism, default is MCAR
        indicator = boolean to indicate which rows to spike in more/less missing for sensitivity analysis
    Return:
        Returns dataframe with additional missing spiked in
    """
    columns = dataframe.drop(columns = ['PUF_VITAL_STATUS','DX_LASTCONTACT_DEATH_MONTHS']).columns
    if mechanism == 'MCAR':
        n_samples = round(len(dataframe)*percent) # number of samples to make missing
        for col in columns:
            # select indices to replace with nan
            index = dataframe[col].loc[dataframe[col].notna()].sample(n_samples).index
            dataframe.loc[index, col] = np.nan
        return dataframe
    elif mechanism == 'MAR':
        n_samples = round(sum(~indicator)*percent/2) 
        n_samples_stage4 = round(sum(indicator)*percent)
        for col in columns:
            index = dataframe[col].loc[(dataframe[col].notna() & indicator)].sample(n_samples_stage4).index
            dataframe.loc[index, col] = np.nan
            index = dataframe[col].loc[(dataframe[col].notna() & ~indicator)].sample(n_samples).index
            dataframe.loc[index, col] = np.nan
        return dataframe



# Function for performing column substitution as control
def impute_simple(data_miss, varnames, categorical, method='mean'):
    """
    Description:
        This function uses a simple mean/median/mode subsitution for imputation.
        Mean/median/mode is calculated per column (variable).
    Param:
        data_miss = dataframe with missing values np.nan
        varnames = list of name of variables for which imputation is being applied
        categorical = boolean indicating if varnames are categorical variables
        method = method for imputation; mean (default), median, or mode
    Return:
        Returns a dataframe where np.nan are replaced by column mean/median/mode
    """
    if method == 'mean':
        imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    elif method == 'median':
        imputer = SimpleImputer(missing_values=np.nan, strategy='median')
    elif method == 'mode':
        imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    #round categoricals
    if categorical == True:
        data_imputed = pd.DataFrame(np.round(imputer.fit_transform(data_miss[varnames])), columns=varnames)
    else:
        data_imputed = pd.DataFrame(imputer.fit_transform(data_miss[varnames]), columns=varnames)
    i = 0
    for var in varnames:
        if categorical == True:
            imputed_value = np.round(imputer.statistics_[i])
        else:
            imputed_value = imputer.statistics_[i]
        i+=1 
    return data_imputed[varnames]


# fit 0-1 scaler on original complete data
scaler = MinMaxScaler()
scaler.fit(df[x[2]])

# run experiments for 10-50% missing save imputed datasets
methods = ['sub', 'mice', 'knn', 'softimpute', 'iterativesvd', 'missforest','midas']
runtimes = pd.DataFrame({'method':methods}) # initialize runtimes table to save runtimes

df_miss = df.copy() # initialize df_miss

for percent_miss in [10, 20, 30, 40, 50]:


    print('\n\nImputing for '+str(percent_miss)+'% missing spike-in:')
    df_miss = spike_missing(df_miss) # spike additional 10% missing
    df_miss.to_pickle(RESULT_FOLDER+str(percent_miss)+'_df_miss.pkl', protocol = 4) #df_miss is save before scaling and encoding
    df_i = df_miss.copy() # initialize df_i

    # SIMPLE SUBSTITUTION -- no need to scale or encode for this one
    print('\n\nStarting substitution:')
    start = time.time()
    df_i[x[2]] = impute_simple(df_miss, x[2], categorical=False, method='median')
    df_i[x[0]+x[1]] = impute_simple(df_miss, x[0]+x[1], categorical=True, method='mode')
    runtime = time.time() - start
    print('Run time for substitution:', runtime)
    runtimes.loc[runtimes['method']=='sub', str(percent_miss)] = runtime
    df_i.to_pickle(RESULT_FOLDER+str(percent_miss)+'_df_sub.pkl', protocol = 4)

    # one-hot-encode df_miss and scale continuous variables within the missing dataframe
    encoded = process_data.one_hot_encode(df_miss, x)
    df_miss = encoded[0]
    encoded_cols = encoded[1] #this is a list of list
    scaled = scaler.transform(df_miss[x[2]])
    df_miss[x[2]] = pd.DataFrame(scaled, columns = x[2])


    # MICE
    print('\n\nStarting MICE:')
    start = time.time()
    imputer = IterativeImputer(min_value=0, max_value=1)
    df_i = pd.DataFrame(imputer.fit_transform(df_miss), columns=df_miss.columns)
    runtime = time.time() - start
    print('Run time:', runtime)
    runtimes.loc[runtimes['method']=='mice', str(percent_miss)] = runtime
    # decode and unscale
    df_i = process_data.ohe_decode(df_i, x)
    unscaled = scaler.inverse_transform(df_i[x[2]])
    df_i[x[2]] = pd.DataFrame(unscaled, columns = x[2])
    df_i.to_pickle(RESULT_FOLDER+str(percent_miss)+'_df_mice.pkl', protocol = 4)


    # KNN # need pip install scikit-learn-0.22.1
    print('\n\nStarting KNN:')
    start = time.time()
    imputer = KNNImputer() # default neighbors is 5
    df_i = pd.DataFrame(imputer.fit_transform(df_miss), columns=df_miss.columns)
    runtime = time.time() - start
    print('Run time:', runtime)
    runtimes.loc[runtimes['method']=='knn', str(percent_miss)] = runtime
    # decode and unscale
    df_i = process_data.ohe_decode(df_i, x)
    unscaled = scaler.inverse_transform(df_i[x[2]])
    df_i[x[2]] = pd.DataFrame(unscaled, columns = x[2])
    df_i.to_pickle(RESULT_FOLDER+str(percent_miss)+'_df_knn.pkl', protocol = 4)


    # SoftImpute
    print('\n\nStarting SoftImpute:')
    start = time.time()
    imputer = SoftImpute(min_value=0, max_value=1)
    df_i = pd.DataFrame(imputer.fit_transform(df_miss), columns=df_miss.columns)
    runtime = time.time() - start
    print('Run time:', runtime)
    runtimes.loc[runtimes['method']=='softimpute', str(percent_miss)] = runtime
    # unscale and decode
    df_i = process_data.ohe_decode(df_i, x)
    unscaled = scaler.inverse_transform(df_i[x[2]])
    df_i[x[2]] = pd.DataFrame(unscaled, columns = x[2])
    df_i.to_pickle(RESULT_FOLDER+str(percent_miss)+'_df_softimpute.pkl', protocol = 4)


    # IterativeSVD
    print('\n\nStarting IterativeSVD:')
    start = time.time()
    imputer = IterativeSVD(min_value=0, max_value=1)
    df_i = pd.DataFrame(imputer.fit_transform(df_miss), columns=df_miss.columns)
    runtime = time.time() - start
    print('Run time:', runtime)
    runtimes.loc[runtimes['method']=='iterativesvd', str(percent_miss)] = runtime
    # unscale and decode
    df_i = process_data.ohe_decode(df_i, x)
    unscaled = scaler.inverse_transform(df_i[x[2]])
    df_i[x[2]] = pd.DataFrame(unscaled, columns = x[2])
    df_i.to_pickle(RESULT_FOLDER+str(percent_miss)+'_df_iterativesvd.pkl', protocol = 4)


    # MissForest
    # reload df_miss as df_miss2 because missforest is not one-hot-encoded
    df_miss2 = pd.read_pickle(RESULT_FOLDER+str(percent_miss)+'_df_miss.pkl')
    df_miss2[x[2]] = pd.DataFrame(scaler.transform(df_miss2[x[2]]), columns = x[2])
    i_cat = [] # initial list to store index of catgorical columns
    for col in x[0]+x[1]:
        i = df_miss2.columns.get_loc(col)
        i_cat.append(i)
    print('\n\nStarting MissForest:')
    start = time.time()
    imputer = MissForest()
    imputer.fit(df_miss2, cat_vars=i_cat)
    df_i = pd.DataFrame(imputer.transform(df_miss2), columns=df_miss2.columns)
    runtime = time.time() - start
    print('Run time:', runtime)
    runtimes.loc[runtimes['method']=='missforest', str(percent_miss)] = runtime
    # unscale (no need to decode for missforest)
    unscaled = scaler.inverse_transform(df_i[x[2]])
    df_i[x[2]] = pd.DataFrame(unscaled, columns = x[2])
    df_i.to_pickle(RESULT_FOLDER+str(percent_miss)+'_df_missforest.pkl', protocol = 4)

    # Denoising autoencoder Midas # need sklearn scikit_learn==0.24.2
    # os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    # random.seed(seed)
    # np.random.seed(seed)
    # tf.random.set_seed(seed)
    print('\n\nStarting denoising autoencoder:')
    start = time.time()
    # run midas imputation
    imputer = Midas(seed=seed) # set seed or else generates an error
    imputer.build_model(df_miss, softmax_columns = encoded_cols)
    imputer.train_model() # default 100 epochs
    imputations = imputer.generate_samples(m=1).output_list
    df_i = imputations[0]
    runtime = time.time() - start
    print('Run time:', runtime)
    runtimes.loc[runtimes['method']=='midas', str(percent_miss)] = runtime

    # unscale and decode
    df_i = process_data.ohe_decode(df_i, x)
    unscaled = scaler.inverse_transform(df_i[x[2]])
    df_i[x[2]] = pd.DataFrame(unscaled, columns = x[2])
    df_i.to_pickle(RESULT_FOLDER+str(percent_miss)+'_df_midas.pkl', protocol = 4)

    # reload unscaled and un-encoded df_miss for next loop
    df_miss = pd.read_pickle(RESULT_FOLDER+str(percent_miss)+'_df_miss.pkl')


# save and print runtimes
runtimes.to_csv(RESULT_FOLDER+'runtimes.csv', index=False)
print(runtimes)
