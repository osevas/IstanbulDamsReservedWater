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
import datetime

import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import SimpleExpSmoothing






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

df_origin=pd.DataFrame(columns=['Date','GeneralDamOccupancyRate','GeneralDamReservedWater'])

# Writing data into dataframe
for child in records:
    df_origin=df_origin.append({'Date':child.get('DATE'),'GeneralDamOccupancyRate':child.get('GENERAL_DAM_OCCUPANCY_RATE'),'GeneralDamReservedWater':child.get('GENERAL_DAM_RESERVED_WATER')},ignore_index=True)

# %%--------------------------------------------
# Data Organization
# --------------------------------------------
# Making Date column index
df=df_origin.copy()

print('Data acquired from '+data_dict.get('help'))
print('Sql code: '+result.get('sql'))
print('First record date: '+df.loc[0,'Date'])
print('Last record date: '+df.iloc[-1,0])

df['Date']=pd.to_datetime(df['Date'],yearfirst=True,format='%Y-%m-%d')
df.set_index(df['Date'],drop=True,inplace=True)
df.drop(columns='Date',inplace=True)

# Drop NAs
#df.dropna(axis=0,inplace=True)

df=df.asfreq(freq='d',method='ffill')

# for simplicaity
reservedWater='GeneralDamReservedWater'



# %%--------------------------------------------
# Initial Data Visualization
# --------------------------------------------
plt.figure(figsize=(20,10))
df[reservedWater].plot()
plt.title('Reserved Water in Dams')
plt.ylabel('[M m^3]')
plt.grid(b=True,which='both',axis='both')

# MOVING AVERAGES
# Weekly
df.rolling(window=7).mean()[reservedWater].plot()

# Montly
df.rolling(window=30).mean()[reservedWater].plot()

# %%--------------------------------------------
# Time Series Analysis
# --------------------------------------------

# %%--------------------------------------------
# ETS Decomposition
# --------------------------------------------
# Decomposing to Trend-Seasonal-Residual
sd=seasonal_decompose(df[reservedWater],model='additive')
sd.trend
sd.seasonal
sd.resid
sd.plot()

# %%--------------------------------------------
# Simple Exponential Smoothing
# --------------------------------------------
model=SimpleExpSmoothing(df[reservedWater])
fit_model=model.fit(optimized=True,use_brute=True)
fit_model.fittedvalues.plot()

plt.figure(figsize=(12,6))
plt.plot(df.index.values,fit_model.fittedvalues.shift(100))
plt.plot(df.index.values,df[reservedWater])
plt.legend()
plt.show()



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

