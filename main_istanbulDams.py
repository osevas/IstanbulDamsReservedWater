# Subject: Time series analysis on reserved water in istanbul
# Author: Onur Sevket Aslan
# Date: 2020-12-24
# Evaluation: I will try various models.  The most accurate model will be deployed.

# %% ----------------------------------------- 
# Libraries
# --------------------------------------------
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from importlib import reload
import urllib
import json




# %% ----------------------------------------- 
# Data Ingestion
# --------------------------------------------
url = 'https://data.ibb.gov.tr/api/3/action/datastore_search_sql?sql=SELECT%20*%20from%20%22b68cbdb0-9bf5-474c-91c4-9256c07c4bdf%22'  
fileobj = urllib.request.urlopen(url)
data_string=fileobj.read() # converts HTTPResponse to string
data_dict=json.loads(data_string) #converts string to dictionary
# Records key stores the necessary data
result=data_dict.get('result')
records=result.get('records')

df=pd.DataFrame(columns=['Date','GeneralDamOccupancyRate','GeneralDamReservedWater'])

# Writing data into dataframe
for child in records:
    df=df.append({'Date':child.get('DATE'),'GeneralDamOccupancyRate':child.get('GENERAL_DAM_OCCUPANCY_RATE'),'GeneralDamReservedWater':child.get('GENERAL_DAM_RESERVED_WATER')},ignore_index=True)


# --------------------------------------------
# Exploratory data analysis
# --------------------------------------------




# %%--------------------------------------------
# Feature Analysis
# --------------------------------------------


#%% --------------------------------------------
# Feature Analysis for continuous features (type=2)
# ----------------------------------------------

# ----------------------------------------------

#%% --------------------------------------------
# Feature Analysis for columns with type=3
# ----------------------------------------------




#%% --------------------------------------------
# Outlier removal from features
# ----------------------------------------------


#%% --------------------------------------------
# Train-valid-split & Scaling
# --------------------------------------------


#%%
# Scaling


#%% --------------------------------------------
# Fit & Prediction of Linear Regression
# --------------------------------------------


#%% --------------------------------------------
# Comparison of models
# --------------------------------------------

#%% --------------------------------------------
# Residual plot
# --------------------------------------------


#************RUN THE MODEL ON THE TEST DATA**************************

