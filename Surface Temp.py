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


# In[67]:


df=pd.read_csv('D:\\Dissertation AM\\Dissertation\\Dissertation\\outcomes to measure climate change\\surface temperature\\GlobalTemperatures.csv') 
#df = df.iloc[:-5]
df


# In[38]:


df.info()


# In[39]:


# Convert 'dt' column to datetime format
df['dt'] = pd.to_datetime(df['dt'])
df


# In[40]:


df.info()


# In[41]:


df.describe()


# In[42]:


# Plot histograms for temperature columns
for column in temperature_columns:
    plt.figure(figsize=(8, 5))
    plt.hist(df[column], bins=20, edgecolor='black', alpha=0.7)
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.title(f'Histogram of {column}')
    plt.grid(True)
    plt.show()


# In[10]:


# Perform mean imputation for the temperature columns
df[temperature_columns] = df[temperature_columns].fillna(df[temperature_columns].mean())

df


# In[68]:


ts=df.groupby(["LandAverageTemperature"])["LandAverageTemperature"].sum()
ts.astype('float')
plt.figure(figsize=(14,8))
plt.title('Global Average Absolute Sea Level Change')
plt.xlabel('Time')
plt.ylabel('Sea Level Change')
plt.plot(df);


# In[43]:


# Plot Land Average Temperature against Date
plt.figure(figsize=(10, 6))
plt.plot(df['dt'], df['LandAverageTemperature'], marker='o')
plt.xlabel('Date')
plt.ylabel('Land Average Temperature')
plt.title('Land Average Temperature Over Time')
plt.grid(True)
plt.show()


# In[44]:


# Define the time frame you want to analyze
start_date = '1850-01-01'
end_date = '2015-12-31'

# Filter the data within the specified time frame
filtered_df = df[(df['dt'] >= start_date) & (df['dt'] <= end_date)]

# Calculate the average temperature for each year
average_temp_per_year = filtered_df.groupby(filtered_df['dt'].dt.year)['LandAverageTemperature'].mean()

# Create a plot for average temperature over the time frame
plt.figure(figsize=(10, 6))
plt.plot(average_temp_per_year.index, average_temp_per_year.values, marker='o')
plt.xlabel('Year')
plt.ylabel('Average Temperature')
plt.title(f'Average Land Average Temperature ({start_date} to {end_date})')
plt.grid(True)
plt.show()


# In[45]:


# Define the time frame you want to analyze
start_date = '2000-01-01'
end_date = '2002-01-01'

# Filter the data within the specified time frame
filtered_df = df[(df['dt'] >= start_date) & (df['dt'] < end_date)]

# Create a line plot for the temperature data with markers
plt.figure(figsize=(10, 6))
plt.plot(filtered_df['dt'], filtered_df['LandAverageTemperature'], marker='o', linestyle='-', alpha=0.7)
plt.xlabel('Date')
plt.ylabel('Land Average Temperature')
plt.title(f'Land Average Temperature ({start_date} to {end_date})')
plt.grid(True)
plt.show()


# In[46]:


plt.figure(figsize=(14,6))
plt.plot(df['LandAverageTemperature'].rolling(window=12,center=False).mean(),label='Rolling Mean');
plt.plot(df['LandAverageTemperature'].rolling(window=12,center=False).std(),label='Rolling Deviation');
plt.legend() 


# In[75]:


# Additive model
res = sm.tsa.seasonal_decompose(ts.values, period=12, model="additive")
plt.figure(figsize=(50,50))
fig = res.plot()
fig.show()


# In[27]:


print(res)


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


# In[48]:


df=df[['dt','LandAverageTemperature','LandAverageTemperatureUncertainty'] ]


# In[49]:


df


# In[50]:


from prophet import Prophet
data=df
df.rename(columns={'Time': 'ds', 'LandAverageTemperature': 'y', 'LandAverageTemperatureUncertainty': 'yhat'}, inplace=True)  
ts=df
ts.columns=['ds','y','yhat']
model1 = Prophet( yearly_seasonality=True) 
model1.fit(ts)


# In[51]:


ts.head()


# In[52]:


future = model1.make_future_dataframe(periods = 240, freq = 'MS')  
# now lets make the forecasts
forecast = model1.predict(future)
fore=forecast[['ds','yhat', 'yhat_lower', 'yhat_upper']]
fore


# In[53]:


fore['ds'] = pd.to_datetime(fore['ds'])
fore.set_index('ds', inplace=True)   


# In[56]:


observed['ds'] = pd.to_datetime(observed['ds'])
observed.set_index('ds', inplace=True)   
observed 


# In[58]:


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


# In[65]:


# Calculate the mean of 'y' and 'yhat' per year
mean_y_per_year = merged_df.groupby(merged_df['ds'].dt.year)['y'].mean()
mean_yhat_per_year = merged_df.groupby(merged_df['ds'].dt.year)['yhat'].mean()

# Plot the means of 'y' and 'yhat' values per year
plt.figure(figsize=(10, 6))
plt.plot(mean_y_per_year.index, mean_y_per_year.values, marker='o', label='Observed')
plt.plot(mean_yhat_per_year.index, mean_yhat_per_year.values, marker='o', label='Forecasted')
plt.xlabel('Year')
plt.ylabel('Mean Value')
plt.title('Mean Values of y and yhat Per Year')
plt.legend()
plt.grid(True)
plt.show()


# In[59]:


model1.plot(forecast)


# In[66]:


# Define the specific time period you want to plot
start_date = '2015-01-01'
end_date = '2030-12-31'

# Filter the 'forecast' DataFrame for the specific time period
forecast_subset = forecast[(forecast['ds'] >= start_date) & (forecast['ds'] <= end_date)]

# Plot the forecasted values for the specific time period
model1.plot(forecast_subset)
plt.xlabel('Date')
plt.ylabel('Forecasted Value')
plt.title('Forecasted Values for Specific Time Period')
plt.show()


# ### LSTM 

# In[80]:


data_array = np.array(data)

# Reshape the data for LSTM input
sequence_length = 3 # Number of previous time steps to consider for each prediction

X = []
y = []

for i in range(len(data_array) - sequence_length):
    X.append(data_array[i:i+sequence_length])
    y.append(data_array[i+sequence_length])

X = np.array(X)
y = np.array(y)

print("X:")
print(X)
print("y:")
print(y)


# In[81]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Split the data into training and testing sets
split_ratio = 0.8
split_index = int(split_ratio * len(X))

X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# Build the LSTM model
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(sequence_length, 1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# Train the model
model.fit(X_train, y_train, epochs=20, batch_size=1, verbose=1)

# Make predictions
y_pred = model.predict(X_test)

# Inverse transform to get original scale
#y_pred = scaler.inverse_transform(y_pred)
#y_test = scaler.inverse_transform(y_test)


# In[76]:


X_test


# In[82]:


# Calculate metrics
mae = np.mean(np.abs(y_test - y_pred))
mse = np.mean((y_test - y_pred) ** 2)
rmse = np.sqrt(mse)

print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)

# Plot observed vs. predicted values
plt.figure(figsize=(10, 6))
plt.plot(y_test, label='Observed')
plt.plot(y_pred, label='Predicted', linestyle='dashed')
plt.xlabel('Time')
plt.ylabel('Value')
plt.title('Observed vs. Predicted (LSTM)')
plt.legend()
plt.show()


# In[88]:


# Number of future time steps to forecast
forecast_steps = 240

# Initialize an array to store the forecasted values
forecasted_values = []

# Seed the initial input sequence with the last few data points from the test set
input_sequence = X_test[-sequence_length:]

for _ in range(forecast_steps):
    # Reshape the input sequence for the LSTM model
    input_sequence_reshaped = input_sequence #.reshape((1, sequence_length, 1))
    
    # Make a prediction for the next time step
    prediction = model.predict(input_sequence_reshaped)
    
    # Append the prediction to the forecasted values
    forecasted_values.append(prediction[0, 0])
    
    # Update the input sequence by removing the first element and adding the prediction
    input_sequence = np.roll(input_sequence, -1)
    input_sequence[-1] = prediction[0, 0]


# In[93]:


len(forecasted_values)


# In[94]:


# Assuming you have a DataFrame df with columns "ds," "y," and "yhat"
forecast_period = 240  # 20 years * 12 months per year

# Get the last "sequence_length" values from your data
last_sequence = ts['y'].tail(sequence_length).values

# Create future dates for forecasting
forecast_dates = pd.date_range(start=ts['ds'].iloc[-1], periods=forecast_period + 1, freq='MS')[1:]



# In[96]:


# Assuming you have a NumPy array called 'forecast_values'

# Convert the array to a DataFrame
forecast_df = pd.DataFrame({'ds': forecast_dates, 'yhat': forecasted_values})

# Print the forecast DataFrame
print(forecast_df)


# In[98]:


forecast_df


# In[95]:


forecast_dates


# In[101]:


# Plot observed vs. forecasted values
plt.figure(figsize=(10, 6))
plt.plot(ts['ds'], ts['y'], label='Observed')
#plt.plot(forecast_df['ds'], forecast_df['yhat'], label='Forecasted')
plt.xlabel('Date')
plt.ylabel('Value')
plt.title('Observed vs. Forecasted')
plt.legend()
plt.show()


# In[ ]:




