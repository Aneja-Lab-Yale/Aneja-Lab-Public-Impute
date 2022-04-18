## Missing RWD imputation

Research project comparing the efficacy of machine learning methods to impute missing clinical information within registry data for lung cancer patients. The manuscript is currently under submission.

## Datasets
We used the National Cancer Database (NCDB) participant user file in the paper. Our analysis was limited to patients diagnosed in 2014 with complete data in variables of interest.

## Code Flow

- `impute_main.py` Main file
- `process_data.py` Functions for processing data file
- `constants.py` File for defining constants
- `metrics.py` Metrics for calculating differences between complete dataset and dataset with missing info
- `TableFigures.py` Create tables and figures for paper

## How to Run
Install requirements.txt  
Update numpy and sklearn as in update_numpy.txt and update_sklearn.txt  
Create data folder with .dta data and variables.csv
Create result folder
```
python impute_main.py 
python TablesFigures.py
```
