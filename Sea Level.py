#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn.model_selection as ms
import numpy.random as nr
import datetime
from sklearn import feature_selection as fs
from sklearn.ensemble import RandomForestRegressor
import statsmodels.api as sm
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error,  r2_score
from sklearn.linear_model import Ridge


# In[8]:


df=pd.read_csv('D:\\Dissertation AM\\Dissertation\\Dissertation\\outcomes to measure climate change\\sea level\\sea_levels_2015.csv') 
#df = df.iloc[:-5]
df


# In[17]:


#formatting the date column correctly
df.Time=df.Time.apply(lambda x:datetime.datetime.strptime(x, '%Y-%m-%d'))
# check
print(df.info())


# In[19]:


ts=df.groupby(["GMSL"])["GMSL"].sum()
ts.astype('float')
plt.figure(figsize=(14,8))
plt.title('Global Average Absolute Sea Level Change')
plt.xlabel('Time')
plt.ylabel('Sea Level Change')
plt.plot(ts);


# In[32]:


plt.figure(figsize=(14,6))
plt.plot(ts.rolling(window=12,center=False).mean(),label='Rolling Mean');
plt.plot(ts.rolling(window=12,center=False).std(),label='Rolling Deviation');
plt.legend();


# In[22]:


# Additive model
res = sm.tsa.seasonal_decompose(ts.values, period=12, model="additive")
#plt.figure(figsize=(16,12))
fig = res.plot()
#fig.show()


# In[27]:


import statsmodels.api as sm
import warnings
warnings.filterwarnings("ignore")
mod = sm.tsa.statespace.SARIMAX(ts.values,
                                order = (2, 0, 4),
                                seasonal_order = (3, 1, 2, 12),
                                enforce_stationarity = False,
                                enforce_invertibility = False)
results = mod.fit()
results.plot_diagnostics(figsize=(14,12))
plt.show()


# In[33]:


from prophet import Prophet
data=df
df.rename(columns={'Time': 'ds', 'GMSL': 'y', 'GMSL uncertainty': 'yhat'}, inplace=True)  
ts=df
ts.columns=['ds','y','yhat']
model1 = Prophet( yearly_seasonality=True) 
model1.fit(ts)


# In[35]:


ts.head()


# In[48]:


future = model1.make_future_dataframe(periods = 480, freq = 'MS')  
# now lets make the forecasts
forecast = model1.predict(future)
fore=forecast[['ds','yhat', 'yhat_lower', 'yhat_upper']]


# In[49]:


fore


# In[50]:


from sklearn.metrics import mean_absolute_error, mean_squared_error
# Merge observed and forecast data on 'ds' column
observed=df.drop(columns=['yhat'])
merged_df = observed.merge(fore, on='ds', how='inner')

# Calculate metrics
mae = mean_absolute_error(merged_df['y'], merged_df['yhat'])
mse = mean_squared_error(merged_df['y'], merged_df['yhat'])
rmse = np.sqrt(mse)
mape = np.mean(np.abs((merged_df['y'] - merged_df['yhat']) / merged_df['y'])) * 100

print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)


# In[43]:


merged_df


# In[51]:


import matplotlib.pyplot as plt

# Plot observed vs. forecasted values
plt.figure(figsize=(10, 6))
plt.plot(merged_df['ds'], merged_df['y'], label='Observed')
plt.plot(merged_df['ds'], merged_df['yhat'], label='Forecasted')
plt.xlabel('Date')
plt.ylabel('Value')
plt.title('Observed vs. Forecasted')
plt.legend()
plt.show()


# In[52]:


model1.plot(forecast)


# In[ ]:




