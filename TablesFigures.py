# Missing data imputation
# File to show tables and figures
# Aneja Lab | Yale School of Medicine
# Daniel Yang, Miles Hui

import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd

# from sklearn import preprocessing
# from sklearn.impute import SimpleImputer
import seaborn as sns
from sksurv.linear_model import CoxnetSurvivalAnalysis, CoxPHSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse
from sklearn.model_selection import train_test_split
from tableone import TableOne


# import user written modules and constants
from constants import DATA_FOLDER, RESULT_FOLDER, DATA_NAME
from constants import VARIABLES_TYPE
import metrics
from process_data import variable_type, one_hot_encode, split_df

# load data
df = pd.read_stata(DATA_FOLDER + DATA_NAME) # note this dataset has survival columns
df = df.astype(float) # convert all to float
# x is a tuple where x[0]=categorical; x[1]=ordinal; x[2]=continuous
# x[0]+x[1] is treated as categorical in paper
variables = pd.read_csv(VARIABLES_TYPE, header=0)
x = variable_type(df, variables)

methods = ['sub', 'knn', 'softimpute', 'iterativesvd', 'missforest', 'midas']
### Table of overall imputaton performance ###
table1 = pd.DataFrame({'method':methods})


# read in the pkl data files and generate table 1
for percent_miss in [10, 20, 30, 40, 50]:
# read the dataframes corresponding to each percent missing
    df_miss = pd.read_pickle(RESULT_FOLDER+str(percent_miss)+'_df_miss.pkl')
    df_sub = pd.read_pickle(RESULT_FOLDER+str(percent_miss)+'_df_sub.pkl')
    df_knn = pd.read_pickle(RESULT_FOLDER+str(percent_miss)+'_df_knn.pkl')
    # df_mice = pd.read_pickle(FOLDER + str(percent_miss) + '_df_mice.pkl')
    df_softimpute = pd.read_pickle(RESULT_FOLDER+str(percent_miss)+'_df_softimpute.pkl')
    df_iterativesvd = pd.read_pickle(RESULT_FOLDER+str(percent_miss)+'_df_iterativesvd.pkl')
    df_missforest = pd.read_pickle(RESULT_FOLDER+str(percent_miss)+'_df_missforest.pkl')
    df_midas = pd.read_pickle(RESULT_FOLDER+str(percent_miss)+'_df_midas.pkl')
    # store in table1
    table1.loc[table1['method']=='sub', str(percent_miss)+'_pfc'] = metrics.pfc(df, df_miss, df_sub, x)
    table1.loc[table1['method']=='sub', str(percent_miss)+'_rmse'] = metrics.rmse(df, df_miss, df_sub, x)
    table1.loc[table1['method']=='knn', str(percent_miss)+'_pfc'] = metrics.pfc(df, df_miss, df_knn, x)
    table1.loc[table1['method']=='knn', str(percent_miss)+'_rmse'] = metrics.rmse(df, df_miss, df_knn, x)
    # table1.loc[table1['method']=='mice', str(percent_miss) + '_pfc'] = metrics.pfc(df, df_miss, df_mice, x)
    # table1.loc[table1['method']=='mice', str(percent_miss) + '_rmse'] = metrics.rmse(df, df_miss, df_mice, x)
    table1.loc[table1['method']=='softimpute', str(percent_miss)+'_pfc'] = metrics.pfc(df, df_miss, df_softimpute, x)
    table1.loc[table1['method']=='softimpute', str(percent_miss)+'_rmse'] = metrics.rmse(df, df_miss, df_softimpute, x)
    table1.loc[table1['method']=='iterativesvd', str(percent_miss)+'_pfc'] = metrics.pfc(df, df_miss, df_iterativesvd, x)
    table1.loc[table1['method']=='iterativesvd', str(percent_miss)+'_rmse'] = metrics.rmse(df, df_miss, df_iterativesvd, x)
    table1.loc[table1['method']=='missforest', str(percent_miss)+'_pfc'] = metrics.pfc(df, df_miss, df_missforest, x)
    table1.loc[table1['method']=='missforest', str(percent_miss)+'_rmse'] = metrics.rmse(df, df_miss, df_missforest, x)
    table1.loc[table1['method']=='midas', str(percent_miss)+'_pfc'] = metrics.pfc(df, df_miss, df_midas, x)
    table1.loc[table1['method']=='midas', str(percent_miss)+'_rmse'] = metrics.rmse(df, df_miss, df_midas, x)

table1.to_csv(RESULT_FOLDER+'table1.csv', index=False)
print(table1)


### Table of performance by variables ###
table2 = pd.DataFrame({'method':methods})

percent_miss = 20
df_miss = pd.read_pickle(RESULT_FOLDER+str(percent_miss)+'_df_miss.pkl')
df_sub = pd.read_pickle(RESULT_FOLDER+str(percent_miss)+'_df_sub.pkl')
# df_mice = pd.read_pickle(FOLDER+str(percent_miss)+'_df_mice.pkl')
df_knn = pd.read_pickle(RESULT_FOLDER+str(percent_miss)+'_df_knn.pkl')
df_softimpute = pd.read_pickle(RESULT_FOLDER+str(percent_miss)+'_df_softimpute.pkl')
df_iterativesvd = pd.read_pickle(RESULT_FOLDER+str(percent_miss)+'_df_iterativesvd.pkl')
df_missforest = pd.read_pickle(RESULT_FOLDER+str(percent_miss)+'_df_missforest.pkl')
df_midas = pd.read_pickle(RESULT_FOLDER+str(percent_miss)+'_df_midas.pkl')



# outcome variables were not imputed
variables = pd.read_csv(VARIABLES_TYPE, header=0)
x = variable_type(df, variables)
x[0].remove('PUF_VITAL_STATUS')
x[2].remove('DX_LASTCONTACT_DEATH_MONTHS')

for var in x[2]:
    table2.loc[table2['method']=='sub', var] = metrics.rmse(df, df_miss, df_sub, x, [var])
    table2.loc[table2['method']=='knn', var] = metrics.rmse(df, df_miss, df_knn, x, [var])
    # table2.loc[table2['method']=='mice', var] = metrics.rmse(df, df_miss, df_mice, x, [var])
    table2.loc[table2['method']=='softimpute', var] = metrics.rmse(df, df_miss, df_softimpute, x, [var])
    table2.loc[table2['method']=='iterativesvd', var] = metrics.rmse(df, df_miss, df_iterativesvd, x, [var])
    table2.loc[table2['method']=='missforest', var] = metrics.rmse(df, df_miss, df_missforest, x, [var])
    table2.loc[table2['method']=='midas', var] = metrics.rmse(df, df_miss, df_midas, x, [var])
for var in x[0]+x[1]:
    table2.loc[table2['method']=='sub', var] = metrics.pfc(df, df_miss, df_sub, x, [var])
    table2.loc[table2['method']=='knn', var] = metrics.pfc(df, df_miss, df_knn, x, [var])
    # table2.loc[table2['method']=='mice', var] = metrics.pfc(df, df_miss, df_mice, x, [var])
    table2.loc[table2['method']=='softimpute', var] = metrics.pfc(df, df_miss, df_softimpute, x, [var])
    table2.loc[table2['method']=='iterativesvd', var] = metrics.pfc(df, df_miss, df_iterativesvd, x, [var])
    table2.loc[table2['method']=='missforest', var] = metrics.pfc(df, df_miss, df_missforest, x, [var])
    table2.loc[table2['method']=='midas', var] = metrics.pfc(df, df_miss, df_midas, x, [var])

# change format of table2
table2 = table2.T
new_header = table2.iloc[0]
table2 = table2[1:]
table2.columns = new_header

# calculate average percent change from substitution
table2['average_change'] = (table2['sub'] - table2[['knn', 'softimpute', 'iterativesvd', 'missforest', 'midas']].mean(axis=1))/table2['sub']*100
table2['min'] = (table2['sub'] - table2[['knn', 'softimpute', 'iterativesvd', 'missforest', 'midas']].min(axis=1))/table2['sub']*100
table2['max'] = (table2['sub'] - table2[['knn', 'softimpute', 'iterativesvd', 'missforest', 'midas']].max(axis=1))/table2['sub']*100

def com_col(x):
    return str(np.round(x['average_change'], 2)) + '% (' + str(np.round(x['max'], 1)) + '-' +  str(np.round(x['min'], 1)) + ')'

table2['Average change (range)'] = table2.apply(lambda x: com_col(x), axis=1)
table2 = table2.drop(columns= ['average_change', 'min', 'max'])
table2[['sub', 'knn', 'softimpute', 'iterativesvd', 'missforest', 'midas']].round(decimals = 3)
table2.to_csv(RESULT_FOLDER+'table2.csv')
print(table2)


# functioon to load and process data for CPH
def load_cph(method):
    """
    Description:
        Function to load and process data for CPH
    Param:
        method = method name to load dataset
    Return:
        Returns two data frames for training and testing
    """
    if method == 'complete':
        df = pd.read_stata(DATA_FOLDER + DATA_NAME)
    else:
        df = pd.read_pickle(RESULT_FOLDER+ '/20_df_{}.pkl'.format(method))

    # preprocess dataset
    df = df.rename(columns={"DX_LASTCONTACT_DEATH_MONTHS": "Time", "PUF_VITAL_STATUS": "Event"})
    df.Event = 1 - df.Event
    df = df.astype('float32')  # convert all to float

    # x is a tuple where x[0]=categorical; x[1]=ordinal; x[2]=continuous
    # x[0]+x[1] is treated as categorical in paper
    x = variable_type(df, variables)

    # one-hot-encode df
    encoded = one_hot_encode(df, x)
    df = encoded[0]
    
    # specific the baseline categories or categories with <5 observations
    base_columns = ['SEX_1.0', 'INSURANCE_STATUS_0.0', 'RX_SUMM_RADIATION_0.0', 'RX_SUMM_SURGRAD_SEQ_0.0',
    'RX_SUMM_CHEMO_0.0', 'RX_SUMM_SYSTEMIC_SUR_SEQ_0.0', 'RACE_1.0', 'RX_SUMM_SURG_PRIM_SITE_0.0',
    'CDCC_TOTAL_BEST_0.0', 'GRADE_1.0', 'ANALYTIC_STAGE_GROUP_1.0', 'MED_INC_QUAR_16_1.0',
    'TNM_CLIN_T_3.0', 'TNM_CLIN_N_2.0', 'TNM_CLIN_M_2.0']
    few_columns = ['ANALYTIC_STAGE_GROUP_5.0', 'RX_SUMM_SURGRAD_SEQ_6.0','RX_SUMM_RADIATION_3.0',
                  'RX_SUMM_SYSTEMIC_SUR_SEQ_5.0', 'RX_SUMM_SYSTEMIC_SUR_SEQ_6.0']
    
    # drop baseline categories and categories with <=5 observations
    df  = df.drop(columns=base_columns, axis=1)
    df  = df.drop(columns=few_columns, axis=1)
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=50)

    return df_train, df_test


def CPH(df_train, df_test):
    """
    Description:
        Function to calculate c-index and get coefficient of CPH model
    Param:
        df_train = training dataset
        df_test = testing dataset
    Return:
        Returns score: c-index, coef: coefficient for CPH
    """
    # tried changing the l1_ratio around with little change
    cph = CoxPHSurvivalAnalysis(alpha=0.001)
    # cph = CoxnetSurvivalAnalysis()
    x_train, x_test, y_train, y_test = split_df(df_train, df_test)
    x_train = x_train.to_numpy().astype('float64')
    y_train = y_train.to_records(index=False)
    cph.fit(x_train, y_train)

    x_test = x_test.to_numpy().astype('float64')
    y_test = y_test.to_records(index=False)
    ci_cox = concordance_index_censored(y_test['Event'], y_test['Time'], cph.predict(x_test))
    score = np.round(ci_cox[0],3)
    coef = np.array(cph.coef_)
    print('c-index is {}'.format(score))
    return score, coef


# generate table 3
methods = ['complete', 'sub', 'knn', 'softimpute', 'iterativesvd', 'missforest', 'midas']
table3 = pd.DataFrame({'method': methods})
cph_coef = pd.DataFrame()
for method in methods:
    print(method)
    df_train, df_test = load_cph(method)
    score, coef = CPH(df_train, df_test)
    table3.loc[table3['method'] == method, 'C-Index'] = score
    cph_coef[method] = coef
for method in methods:
    table3.loc[table3['method'] == method, 'MAE'] = np.round(mae(cph_coef['complete'], cph_coef[method]),3)
    table3.loc[table3['method'] == method, 'MSE'] = np.round(mse(cph_coef['complete'], cph_coef[method]),3)

table3 = table3[['method', 'MAE', 'MSE', 'C-Index']]
table3 = table3.rename(columns = {'method': 'Method'})
table3['Method'] = ['Complete data', 'Substitution', 'KNN', 'SoftImpute',
                    'IterativeSVD', 'MissForest', 'Autoencoder']

table3.to_csv(RESULT_FOLDER+'table3.csv', index=False)
print(table3)


# show the hazard ratio for non-baseline variables
df_col, _ = load_cph('complete')
df_col = df_col.drop(columns=['Time', 'Event'], axis=1)
hr = np.exp(cph_coef).set_index(df_col.columns)
hr.to_csv(RESULT_FOLDER+'cph_hr.csv')
print(hr)


# Figure 2
# load data and convert to long form
df = pd.read_csv(RESULT_FOLDER+'table1.csv')
df = pd.melt(df, id_vars='method')
df[['percent', 'metric']] = df.variable.str.split('_', expand=True)
df['percent'] = pd.to_numeric(df['percent'])
df.drop(columns=['variable'], inplace = True)
print('Long form data: \n', df)

# split rmse and pfc into 2 separate dataframes for graphing
df_rmse = df[df['metric']=='rmse']
df_pfc = df[df['metric']=='pfc']
sns.set_theme(style='darkgrid')

# combined rmse and pfc lineplot
fig, ax =plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
sns.lineplot(x='percent', y='value', data=df_rmse, marker = 'o', hue='method', ax=ax[0], legend=False)
ax[0].set_xticks([10, 20, 30, 40, 50])
ax[0].set_xlabel('Percent missing')
ax[0].set_ylabel('RMSE')
ax[0].set_title('Continuous')
sns.lineplot(x='percent', y='value', data=df_pfc, marker = 'o', hue='method', ax=ax[1])
ax[1].set_xticks([10, 20, 30, 40, 50])
ax[1].set_xlabel('Percent missing')
ax[1].set_ylabel('PFC')
ax[1].set_title('Categorical')
ax[1].legend(bbox_to_anchor=(1.01, 1), labels = ['Substitution', 'KNN', 'SoftImpute', 'IterativeSVD', 'MissForest', 'Autoencoder'])
fig.tight_layout()
plt.savefig(RESULT_FOLDER+'figure2.png', dpi=300, bbox_inches='tight')
plt.show()


# Figure 3
### Runtimes ###

df = pd.read_csv(RESULT_FOLDER+'runtimes.csv')
df = pd.melt(df, id_vars='method')
df['runtime_row'] = df['value']/50790 # normalize runtime by row count
print('Long form data: \n', df)

fig = sns.barplot(x='variable', y='runtime_row', data=df, hue='method')
plt.xlabel('Percent missing')
plt.ylabel('Seconds per row')
plt.yscale('log')
h, l = fig.get_legend_handles_labels()
labels = ['Substitution','KNN', 'SoftImpute', 'IterativeSVD', 'MissForest', 'Autoencoder']
plt.legend(bbox_to_anchor=(1.01, 1), handles = h, labels = labels)
plt.savefig(RESULT_FOLDER+'runtime.png', dpi=300, bbox_inches='tight')
plt.show()



# print average runtimes
print('Substiution:\n', df[df['method']=='sub'].mean())
print('KNN:\n', df[df['method']=='knn'].mean())
print('SoftImpute:\n', df[df['method']=='softimpute'].mean())
print('IterativeSVD:\n', df[df['method']=='iterativesvd'].mean())
print('MissForest:\n', df[df['method']=='missforest'].mean())
print('Autoencoder:\n', df[df['method']=='midas'].mean())

