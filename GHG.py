#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose 
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, Holt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import LSTM
from keras.layers import RepeatVector
from sklearn.metrics import mean_squared_error
from keras.layers import TimeDistributed
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
import plotly.express as px
import plotly.graph_objects as go
import math
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
import geopandas as gpd
from prophet import Prophet
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_absolute_error, mean_squared_error
import itertools
import warnings
warnings.filterwarnings("ignore")


# In[2]:


df=pd.read_csv('D:\\Dissertation AM\\Dissertation\\Dissertation\\GHG Emissions\\CCDR_Final.csv') 
df


# In[3]:


df.info()


# In[4]:


#dropping the countries having large number of missing values

countries_to_drop = ["American Samoa", "Andorra", "Aruba", "Bermuda", "British Virgin Islands","Cayman Islands","Channel Islands","Curacao","Faroe Islands","French Polynesia","Gibraltar","Greenland","Guam","Hong Kong,SAR,China","Isle of Man","Kosovo","Macao SAR, China","Micronesia, Fed. Sts.","Monaco","New Caledonia","Northern Mariana Islands","Puerto Rico","San Marino","Sint Maarten (Dutch part)","St. Martin (French part)","Taiwan", "China", "Turks and Caicos Islands","Virgin Islands (U.S.)","West Bank and Gaza" ]
df = df[~df["Country Name"].isin(countries_to_drop)].reset_index(drop=True)


# In[5]:


df.isnull().sum()


# In[6]:


df_sorted = df.sort_values(by=['Country Name','Time'])

df_sorted.head(10)


# In[7]:


grouped_data = df_sorted.groupby('Country Name')
grouped_data.head()


# In[11]:


# Group by 'Country Name' and calculate the mean for each column
grouped_data = df.groupby('Country Name').mean()

# Function to impute missing values with the column's mean
def impute_with_mean(row):
    country_name = row['Country Name']
    for col in df.columns:
        if pd.isna(row[col]):
            row[col] = grouped_data.loc[country_name, col]
    return row

# Apply the imputation function to each row
df_imputed = df.apply(impute_with_mean, axis=1)


# In[12]:


df.fillna(df.median(), inplace=True)


# In[13]:


df.isnull().sum()


# In[11]:


df.head()


# In[14]:


# Add a new column 'Sum' that holds the sum of all columns except CC.GHG.EMSE.WA and CC.GHG.EMSE.LU
df['Energy'] = df.drop(columns=['CC.GHG.EMSE.WA', 'CC.GHG.EMSE.LU']).sum(axis=1)

# Select the desired columns for the new dataset
df = df[['Time', 'Country Name', 'Energy', 'CC.GHG.EMSE.WA', 'CC.GHG.EMSE.LU']]
df.rename(columns={'CC.GHG.EMSE.WA': 'Waste', 'CC.GHG.EMSE.LU': 'Agriculture'}, inplace=True)
df


# In[12]:


# Plot histograms for temperature columns
for column in df.columns:
    plt.figure(figsize=(8, 5))
    plt.hist(df[column], bins=20, edgecolor='black', alpha=0.7)
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.title(f'Histogram of {column}')
    plt.grid(True)
    plt.show()


# In[13]:


def lstm(data):
    data_array = np.array(data)

   
    sequence_length = 1 # Number of previous time steps to consider for each prediction

    X = []
    y = []

    for i in range(len(data_array) - sequence_length):
        X.append(data_array[i:i+sequence_length])
        y.append(data_array[i+sequence_length])

    X = np.array(X)
    y = np.array(y)
    # Split the data into training and testing 
    split_ratio = 0.8
    split_index = int(split_ratio * len(X))

    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    # LSTM model
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(sequence_length, 1)))
    model.add(Dense(1))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')

    # Train the model
    model.fit(X_train, y_train, epochs=50, batch_size=1, verbose=1)

    # Make predictions
    y_pred = model.predict(X_test)
    
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


# In[76]:


c=0
for country in df['Country Name'].unique():
    # Filter data for the current country
    country_data = df[df['Country Name'] == "United Kingdom"]
    print(country)
    energy=country_data['Waste'] 
    lstm(energy)  
    break 


# In[14]:


def exponential_smoothing(data):
    data_array = np.array(data)

    # Split the data into training and testing sets
    split_ratio = 0.8
    split_index = int(split_ratio * len(data_array))

    data_train, data_test = data_array[:split_index], data_array[split_index:]

    # Initialize the Holt-Winters Exponential Smoothing model
    model = ExponentialSmoothing(data_train, seasonal='add', seasonal_periods=2)

    # Fit the model to the training data
    model_fit = model.fit()

    # Make predictions on the test data
    forecast = model_fit.forecast(steps=len(data_test))  # Forecast for the length of the test data

    mae = np.mean(np.abs(data_test - forecast))
    mse = np.mean((data_test - forecast) ** 2)
    rmse = np.sqrt(mse)

    print("Mean Absolute Error:", mae)
    print("Mean Squared Error:", mse)
    print("Root Mean Squared Error:", rmse)

    # Plot observed data and forecast
    plt.figure(figsize=(10, 6))
    plt.plot(data_test, label='Observed')
    plt.plot(forecast, label='Forecast', linestyle='dashed')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title('Observed vs. Forecast (Exponential Smoothing)')
    plt.legend()
    plt.show()


# In[70]:


for country in df['Country Name'].unique():
    # Filter data for the current country
    country_data = df[df['Country Name'] == "United States"]
    energy=country_data['Waste']
    exponential_smoothing(energy) 
    break  


# In[15]:


def arima(data):
    
    data_array = np.array(data)

    # Reshape the data for ARIMA input
    sequence_length = 1  # Number of previous time steps to consider for each prediction

    X = []
    y = []

    for i in range(len(data_array) - sequence_length):
        X.append(data_array[i:i+sequence_length])
        y.append(data_array[i+sequence_length])

    X = np.array(X)
    y = np.array(y)

    # Split the data into training and testing sets
    split_ratio = 0.8
    split_index = int(split_ratio * len(X))

    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    # Build the ARIMA model
    order = (2, 2, 0)  # (p, d, q)
    model = ARIMA(y_train, order=order)
    model_fit = model.fit()

    # Make predictions
    y_pred = model_fit.predict(start=len(y_train), end=len(y_train) + len(y_test) - 1, typ='levels')

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
    plt.title('Observed vs. Predicted (ARIMA)')
    plt.legend()
    plt.show()


# In[77]:


for country in df['Country Name'].unique():
    # Filter data for the current country
    country_data = df[df['Country Name'] == "United Kingdom"]
    energy=country_data['Waste']
    arima(energy)    
    break 


# In[16]:


def sarima(data):
    data_array = np.array(data)

    # Reshape the data for SARIMA input
    sequence_length = 1  # Number of previous time steps to consider for each prediction

    X = []
    y = []

    for i in range(len(data_array) - sequence_length):
        X.append(data_array[i:i+sequence_length])
        y.append(data_array[i+sequence_length])

    X = np.array(X)
    y = np.array(y)

    # Split the data into training and testing sets
    split_ratio = 0.8
    split_index = int(split_ratio * len(X))

    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    # Build the SARIMA model
    order = (3 , 2 , 1)  # (p, d, q)
    seasonal_order = (3, 3, 0, 12)  # (P, D, Q, S)
    model = SARIMAX(y_train, order=order, seasonal_order=seasonal_order)
    model_fit = model.fit()

    # Make predictions
    y_pred = model_fit.predict(start=len(y_train), end=len(y_train) + len(y_test) - 1)

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
    plt.title('Observed vs. Predicted (SARIMA)')
    plt.legend()
    plt.show()


# In[78]:


for country in df['Country Name'].unique():
    # Filter data for the current country
    country_data = df[df['Country Name'] == "United Kingdom"]
    energy=country_data['Waste']
    sarima(energy)
    break  


# In[17]:


def ar(data):
    data_array = np.array(data)

    # Reshape the data for AR input
    sequence_length = 1  # Number of previous time steps to consider for each prediction

    X = []
    y = []

    for i in range(len(data_array) - sequence_length):
        X.append(data_array[i:i+sequence_length])
        y.append(data_array[i+sequence_length])

    X = np.array(X)
    y = np.array(y)

    # Split the data into training and testing sets
    split_ratio = 0.8
    split_index = int(split_ratio * len(X))

    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    # Build the AutoRegressive (AR) model
    lags = 1  # Number of lagged observations to include in the model
    model = AutoReg(y_train, lags=lags)
    model_fit = model.fit()

    # Make predictions
    y_pred = model_fit.predict(start=len(y_train), end=len(y_train) + len(y_test) - 1)

    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
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
    plt.title('Observed vs. Predicted (AR)')
    plt.legend()
    plt.show()


# In[79]:


for country in df['Country Name'].unique():
    # Filter data for the current country
    country_data = df[df['Country Name'] == "United Kingdom"]
    energy=country_data['Waste']  
    ar(energy) 
    break 


# In[43]:


import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# List of models to try
models_to_try = ['ARIMA', 'SARIMA', 'LSTM', 'AR']  # Add more models if needed

# Store results for each model
results = {}
best_model_info = {'model': None, 'parameter': None, 'rmse': float('inf')}
# Iterate over unique countries
for country in df['Country Name'].unique():
    country_data = df[df['Country Name'] == country]
    
    # Extract time series data for the current country
    time_series_data = country_data['Energy'].values
    
    # Preprocess the data if needed
    
    # Iterate over models
    for model_name in models_to_try:
        model_rmse = []
        
        # AR and LSTM specific settings
        if model_name == 'AR':
            max_lags = 10  # Adjust as needed
        elif model_name == 'LSTM':
            sequence_length = 10  # Adjust as needed
        
        # Iterate over different orders/parameters
        for parameter in range(3):  # Replace with your range
            try:
                if model_name == 'ARIMA':
                    model = ARIMA(time_series_data, order=(parameter, 0, parameter))
                elif model_name == 'SARIMA':
                    model = SARIMAX(time_series_data, order=(parameter, 0, parameter), seasonal_order=(1, 1, 1, 12))
                elif model_name == 'LSTM':
                    # Build and train your LSTM model here
                    model = Sequential()
                    model.add(LSTM(50, activation='relu', input_shape=(sequence_length, 1)))
                    model.add(Dense(1))
                    model.compile(optimizer='adam', loss='mse')
                    # Train the model and get predictions
                elif model_name == 'AR':
                    # Build and train your AR model here
                    # Use up to 'max_lags' lagged observations
                    # Get predictions and calculate RMSE
                    pass
                
                # Calculate RMSE and store
                if model_name == 'LSTM':
                    # Calculate RMSE for LSTM predictions
                    pass
                elif model_name == 'AR':
                    # Calculate RMSE for AR model predictions
                    pass
                else:
                    model_fit = model.fit()
                    y_pred = model_fit.predict()
                    #rmse = np.sqrt(mean_squared_error(time_series_data, y_pred))
                    rmse = mean_absolute_error(time_series_data, y_pred)
                    model_rmse.append(rmse)
            except:
                continue
        
        if model_rmse:
            #results[f"{model_name}"] = np.mean(model_rmse)
            avg_rmse = np.mean(model_rmse)
            if avg_rmse < best_model_info['rmse']:
                best_model_info['model'] = model_name
                best_model_info['parameter'] = parameter
                best_model_info['rmse'] = avg_rmse

# Compare model performance
#best_model = min(results, key=results.get)
#print("Best performing model:", best_model, "with average RMSE:", results[best_model])
print("Best performing model:", best_model_info['model'], "with parameter:", best_model_info['parameter'], "and average MAE:", best_model_info['rmse'])


# In[18]:


def double_exponential_smoothing(df, alpha, beta, n_forecast):
    n = len(df)
    level = np.zeros(n + n_forecast)
    trend = np.zeros(n + n_forecast)
    smoothed = np.zeros(n + n_forecast)
    forecast_errors = []
    # Initialization
    level[0] = df.iloc[0]
    trend[0] = df.iloc[1] - df.iloc[0]
    smoothed[0] = df.iloc[0]

    for i in range(1, n + n_forecast):
        if i < n:
            # Double Exponential Smoothing equations for observed data
            level[i] = alpha * df.iloc[i] + (1 - alpha) * (level[i-1] + trend[i-1])
            trend[i] = beta * (level[i] - level[i-1]) + (1 - beta) * trend[i-1]
            smoothed[i] = level[i] + trend[i]
        else:
            # Forecasting equation for future data
            level[i] = smoothed[i-1]
            trend[i] = trend[i-1]
            smoothed[i] = level[i] + trend[i]
            # Calculate forecast error for future data
            forecast_errors.append(smoothed[i] - df.iloc[-1])

    rmse = np.sqrt(mean_squared_error(df.iloc[-n_forecast:], forecast_errors))
    
    return smoothed[-n_forecast:], rmse

# Example usage
alpha = 0.2  # Smoothing factor for the level (0 <= alpha <= 1)
beta = 0.1   # Smoothing factor for the trend (0 <= beta <= 1)
n_forecast = 20  # Number of years to forecast

# Group by 'Country Name' and apply DES forecast for each country
forecasts_and_rmse = df.groupby('Country Name')['Energy'].apply(
    lambda group: double_exponential_smoothing(group, alpha, beta, n_forecast)
).apply(pd.Series)

# Unpack forecasts and RMSE
forecasts = forecasts_and_rmse[0]
rmse_values_des_energy = forecasts_and_rmse[1]

# Creating the forecast DataFrame with years as index and country names as columns
forecast_years = df['Time'].max() + np.arange(1, n_forecast + 1)
forecast_df_des_energy = pd.DataFrame(np.array(forecasts.tolist()).T, index=forecast_years, columns=forecasts.index)

forecast_df_des_energy


# In[14]:


# Group by 'Country Name' and apply DES forecast for each country
forecasts_and_rmse = df.groupby('Country Name')['Waste'].apply(
    lambda group: double_exponential_smoothing(group, alpha, beta, n_forecast)
).apply(pd.Series)

# Unpack forecasts and RMSE
forecasts = forecasts_and_rmse[0]
rmse_values_des_waste = forecasts_and_rmse[1]

# Creating the forecast DataFrame with years as index and country names as columns
forecast_years = df['Time'].max() + np.arange(1, n_forecast + 1)
forecast_df_des_waste = pd.DataFrame(np.array(forecasts.tolist()).T, index=forecast_years, columns=forecasts.index)

forecast_df_des_waste


# In[15]:


# Group by 'Country Name' and apply DES forecast for each country
forecasts_and_rmse = df.groupby('Country Name')['Agriculture'].apply(
    lambda group: double_exponential_smoothing(group, alpha, beta, n_forecast)
).apply(pd.Series)

# Unpack forecasts and RMSE
forecasts = forecasts_and_rmse[0]
rmse_values_des_agriculture = forecasts_and_rmse[1]  

# Creating the forecast DataFrame with years as index and country names as columns  
forecast_years = df['Time'].max() + np.arange(1, n_forecast + 1)
forecast_df_des_agriculture = pd.DataFrame(np.array(forecasts.tolist()).T, index=forecast_years, columns=forecasts.index)

forecast_df_des_agriculture 


# In[16]:


forecast_df_des_waste.T


# In[24]:


# Find the top 5 countries with the highest forecasted values
top_countries_energy = forecast_df_des_energy.max().nlargest(5).index.tolist()

# Find the least 5 countries with the lowest forecasted values
least_countries_energy  = forecast_df_des_energy.min().nsmallest(5).index.tolist()

print("Top 5 countries with highest forecasted values for energy with DES:")
print(top_countries_energy)

print("\nLeast 5 countries with lowest forecasted values for energy with DES:")
print(least_countries_energy)


# In[19]:


# Create a DataFrame for Plotly Express
forecast_long_energy = forecast_df_des_energy.reset_index().melt(id_vars='index', var_name='Country', value_name='Forecasted Value')
# Replace country names
country_name_mapping = {
    'Russian Federation': 'Russia',
    'Iran, Islamic Rep.': 'Iran'  
}
forecast_long_energy['Country'] = forecast_long_energy['Country'].replace(country_name_mapping)

# Load the world shapefile (using GeoJSON data)
world_shapefile = px.data.gapminder()

# Merge the forecast data with the world shapefile
merged_data = world_shapefile.merge(forecast_long_energy, left_on='country', right_on='Country')

# Create an interactive choropleth map using Plotly Express
fig = px.choropleth(merged_data, 
                    locations='iso_alpha',
                    color='Forecasted Value',
                    hover_name='country',
                    color_continuous_scale='RdYlGn_r',  # Red-Yellow-Green color scale
                    title='Forecasted Values of GHG emission from Energy Sector by Country',
                    labels={'Forecasted Value': 'Forecasted Value'})

# Show the interactive map
fig.show()


# In[17]:


# Find the top 5 countries with the highest forecasted values
top_countries_waste = forecast_df_des_waste.max().nlargest(5).index.tolist()

# Find the least 5 countries with the lowest forecasted values
least_countries_waste  = forecast_df_des_waste.min().nsmallest(5).index.tolist()

print("Top 5 countries with highest forecasted values:")
print(top_countries_waste) 

print("\nLeast 5 countries with lowest forecasted values:")
print(least_countries_waste)


# In[21]:



forecast_long_waste = forecast_df_des_waste.reset_index().melt(id_vars='index', var_name='Country', value_name='Forecasted Value')
# Replace country names
country_name_mapping = {
    'Russian Federation': 'Russia',  
    'Iran, Islamic Rep.': 'Iran', 
    'Congo, Dem. Rep.':'Democratic Republic of the Congo'
}
forecast_long_waste['Country'] = forecast_long_waste['Country'].replace(country_name_mapping)

# Load the world shapefile (using GeoJSON data)
world_shapefile = px.data.gapminder()

# Merge the forecast data with the world shapefile
merged_data = world_shapefile.merge(forecast_long_waste, left_on='country', right_on='Country')

fig = px.choropleth(merged_data, 
                    locations='iso_alpha',
                    color='Forecasted Value',
                    hover_name='country',
                    color_continuous_scale='RdYlGn_r',  # Red-Yellow-Green color scale
                    title='Forecasted Values of GHG emission from Waste Sector by Country',
                    labels={'Forecasted Value': 'Forecasted Value'})

# Show the interactive map
fig.show()


# In[22]:


# Find the top 5 countries with the highest forecasted values
top_countries_agriculture = forecast_df_des_agriculture.max().nlargest(5).index.tolist()

# Find the least 5 countries with the lowest forecasted values
least_countries_agriculture  = forecast_df_des_agriculture.min().nsmallest(5).index.tolist() 

print("Top 5 countries with highest forecasted values:")  
print(top_countries_agriculture) 

print("\nLeast 5 countries with lowest forecasted values:") 
print(least_countries_agriculture)


# In[23]:


# Create a DataFrame for Plotly Express
forecast_long_agriculture = forecast_df_des_agriculture.reset_index().melt(id_vars='index', var_name='Country', value_name='Forecasted Value')
# Replace country names
country_name_mapping = {
    'Russian Federation': 'Russia',  
    'Iran, Islamic Rep.': 'Iran', 
    'Congo, Dem. Rep.':'Democratic Republic of the Congo'
}
forecast_long_agriculture['Country'] = forecast_long_agriculture['Country'].replace(country_name_mapping)

# Load the world shapefile (using GeoJSON data)
world_shapefile = px.data.gapminder()

# Merge the forecast data with the world shapefile
merged_data = world_shapefile.merge(forecast_long_agriculture, left_on='country', right_on='Country')

# Create an interactive choropleth map using Plotly Express
fig = px.choropleth(merged_data, 
                    locations='iso_alpha',
                    color='Forecasted Value',
                    hover_name='country',
                    color_continuous_scale='RdYlGn_r',  # Red-Yellow-Green color scale
                    title='Forecasted Values of GHG emission from Agriculture Sector by Country',
                    labels={'Forecasted Value': 'Forecasted Value'})

# Show the interactive map
fig.show()


# In[25]:


united_states_data = df[df['Country Name'] == 'United Kingdom']

# Perform Double Exponential Smoothing for the United States
forecast_values_e = double_exponential_smoothing(united_states_data['Energy'], alpha, beta, n_forecast)
forecast_values_agri = double_exponential_smoothing(united_states_data['Agriculture'], alpha, beta, n_forecast)
forecast_values_waste = double_exponential_smoothing(united_states_data['Waste'], alpha, beta, n_forecast)
# Creating the forecast DataFrame with years as index
forecast_years = united_states_data['Time'].max() + np.arange(1, n_forecast + 1)
forecast_df_e = pd.DataFrame({'Forecast': forecast_values_e}, index=forecast_years)   
forecast_df_agri = pd.DataFrame({'Forecast': forecast_values_agri}, index=forecast_years)
forecast_df_waste = pd.DataFrame({'Forecast': forecast_values_waste}, index=forecast_years)


# Plotting the historical and forecasted values 
plt.figure(figsize=(12, 6))
plt.plot(united_states_data['Time'], united_states_data['Energy'], label='Historical Energy')  
plt.plot(forecast_df_e.index, forecast_df_e['Forecast'], label='Forecasted Energy')
plt.title("Historical and Forecasted Values") 
plt.xlabel("Year") 
plt.ylabel("Value")
plt.legend()
plt.grid(True)
plt.show()


# In[ ]:


plt.figure(figsize=(12, 6))
plt.plot(united_states_data['Time'], united_states_data['Agriculture'], label='Historical Agriculture')
plt.plot(forecast_df_agri.index, forecast_df_agri['Forecast'], label='Forecasted Agriculture')
plt.plot(united_states_data['Time'], united_states_data['Waste'], label='Historical Waste')
plt.plot(forecast_df_waste.index, forecast_df_waste['Forecast'], label='Forecasted Waste')

plt.title("Historical and Forecasted Values") 
plt.xlabel("Year") 
plt.ylabel("Value")
plt.legend()
plt.grid(True)
plt.show()


# In[ ]:


df


# In[ ]:


last_known_values_e = united_states_data['Energy'].iloc[-1]
last_known_values_agri = united_states_data['Agriculture'].iloc[-1]
last_known_values_waste = united_states_data['Waste'].iloc[-1]

# Calculate the LKV prediction error for each feature
lkv_error_e = forecast_values_e[-1] - last_known_values_e
lkv_error_agri = forecast_values_agri[-1] - last_known_values_agri
lkv_error_waste = forecast_values_waste[-1] - last_known_values_waste

# Display the LKV prediction errors
print("LKV Prediction Error for Energy:", lkv_error_e)
print("LKV Prediction Error for Agriculture:", lkv_error_agri)
print("LKV Prediction Error for Waste:", lkv_error_waste)


# In[132]:


from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


# Extract data for all sectors
energy_data = df['Energy']
waste_data = df['Waste']
agriculture_data = df['Agriculture']

# Augmented Dickey-Fuller test function
def adf_test(df, sector_name):
    result = adfuller(df)
    print(f"ADF Statistic for {sector_name}:", result[0])
    print(f"p-value for {sector_name}:", result[1])
    print(f"Critical Values for {sector_name}:", result[4])

# Perform Augmented Dickey-Fuller test for all sectors
adf_test(energy_data, 'Energy')
adf_test(waste_data, 'Waste')
adf_test(agriculture_data, 'Agriculture')



# In[133]:


# Extract the waste sector data
waste_data = df['Waste']

# Plot ACF and PACF plots  
plt.figure(figsize=(12, 6))

# ACF plot
plt.subplot(121)
plot_acf(waste_data, lags=20, ax=plt.gca(), title="ACF")

# PACF plot
plt.subplot(122)
plot_pacf(waste_data, lags=20, ax=plt.gca(), title="PACF")

plt.tight_layout()
plt.show()


# ### d=1, p=3,q=3 for waste

# In[134]:


# Extract the waste sector data
waste_data = df['Agriculture']

# Plot ACF and PACF plots  
plt.figure(figsize=(12, 6))

# ACF plot
plt.subplot(121)
plot_acf(waste_data, lags=20, ax=plt.gca(), title="ACF")

# PACF plot
plt.subplot(122)
plot_pacf(waste_data, lags=20, ax=plt.gca(), title="PACF")

plt.tight_layout()
plt.show()


# ### d=1, p=3,q=3 for agriculture

# In[135]:


# Extract the waste sector data
waste_data = df['Energy']

# Plot ACF and PACF plots  
plt.figure(figsize=(12, 6))

# ACF plot
plt.subplot(121)
plot_acf(waste_data, lags=20, ax=plt.gca(), title="ACF")

# PACF plot
plt.subplot(122)
plot_pacf(waste_data, lags=20, ax=plt.gca(), title="PACF")

plt.tight_layout()
plt.show()


# ### d=1, p=2,q=2 for waste

# In[65]:


# Extract the list of unique country names
country_names = df['Country Name'].unique()

# Set the ARIMA order (p, d, q)
p, d, q = 3, 1, 3  # Replace with your chosen values

# Dictionary to store forecasts for each country
forecasts_by_country = {}

# Loop through each country
for country in country_names:
    country_data = df[df['Country Name'] == country]['Waste']
    
    # Fit ARIMA model
    model = ARIMA(country_data, order=(p, d, q))
    model_fit = model.fit()
    
    # Forecast for the next 20 years
    forecast_steps = 20
    forecast = model_fit.forecast(steps=forecast_steps)
    
    # Store forecast in the dictionary
    forecasts_by_country[country] = forecast

# Convert the forecasts dictionary to a DataFrame
forecast_df_arima_waste = pd.DataFrame(forecasts_by_country)

# Add years to the DataFrame
years = range(df['Time'].max() + 1, df['Time'].max() + 1 + forecast_steps)
forecast_df_arima_waste['Year'] = years 


# In[66]:


forecast_df_arima_waste


# In[54]:


# Filter the data for the specific years
selected_years = [2011, 2015, 2020, 2025, 2030]
selected_columns = ["Year"] + [country for country in forecast_df_arima_waste if country != "Year"]
selected_data = forecast_df_arima_waste[selected_columns][forecast_df_arima_waste["Year"].isin(selected_years)]

selected_data


# In[22]:


import matplotlib.ticker as ticker
# Calculate the mean across all countries' values for each year
forecast_df_arima_waste["Year"] = forecast_df_arima_waste["Year"].astype(pd.Int64Dtype())
df_mean = forecast_df_arima_waste.drop("Year", axis=1).mean(axis=1)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(forecast_df_arima_waste["Year"].astype(pd.Int64Dtype()), df_mean, marker='o')
plt.xlabel('Year')
plt.ylabel('Mean Value Mt CO2 ')
plt.title('Year-wise Mean Values of Emission from Waste sector Across Countries')
ax = plt.gca()
ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()


# In[24]:


import matplotlib.ticker as ticker

# Calculate the mean across all countries' values for each year
forecast_df_arima_waste["Year"] = forecast_df_arima_waste["Year"].astype(pd.Int64Dtype())
df_mean = forecast_df_arima_waste.drop("Year", axis=1).mean(axis=1)

# Calculate percentage increase
percentage_increase = df_mean.pct_change() * 100

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(forecast_df_arima_waste["Year"].astype(pd.Int64Dtype()), percentage_increase, marker='o')
plt.xlabel('Year')
plt.ylabel('Percentage Increase')
plt.title('Year-wise Percentage Increase in Emission from Waste Sector Across Countries')
ax = plt.gca()
ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()


# In[25]:


import matplotlib.ticker as ticker

# Calculate the mean across all countries' values for each year
forecast_df_arima_waste["Year"] = forecast_df_arima_waste["Year"].astype(pd.Int64Dtype())
df_mean = forecast_df_arima_waste.drop("Year", axis=1).mean(axis=1)

# Calculate percentage increase
percentage_increase = df_mean.pct_change() * 100

# Plotting
plt.figure(figsize=(10, 6))
bars = plt.bar(forecast_df_arima_waste["Year"].astype(pd.Int64Dtype()), percentage_increase)
plt.xlabel('Year')
plt.ylabel('Percentage Increase')
plt.title('Year-wise Percentage Increase in Emission from Waste Sector Across Countries')
ax = plt.gca()
ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()

# Displaying percentage values on top of the bars
for bar, percentage in zip(bars, percentage_increase):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{percentage:.2f}%', ha='center', va='bottom')

plt.show()


# In[23]:


import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import numpy as np

# Assuming forecast_df_arima_waste contains the data

# Calculate the mean across all countries' values for each year
forecast_df_arima_waste["Year"] = forecast_df_arima_waste["Year"].astype(pd.Int64Dtype())
df_mean = forecast_df_arima_waste.drop("Year", axis=1).mean(axis=1)

# Calculate percentage increase
percentage_increase = df_mean.pct_change() * 100

# Plotting
plt.figure(figsize=(10, 6))

# Plotting the line connecting the bar plots
plt.plot(forecast_df_arima_waste["Year"].astype(pd.Int64Dtype()), percentage_increase, marker='o', label='Percentage Increase', color='blue')

# Plotting the bar plot
bars = plt.bar(forecast_df_arima_waste["Year"].astype(pd.Int64Dtype()), percentage_increase, alpha=0.5)

plt.xlabel('Year')
plt.ylabel('Percentage Increase')
plt.title('Year-wise Percentage Increase in Emission from Waste Sector Across Countries')
ax = plt.gca()
ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
plt.xticks(rotation=45)
plt.grid(True)
plt.legend()

# Displaying percentage values on top of the bars
for bar, percentage in zip(bars, percentage_increase):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{percentage:.2f}%', ha='center', va='bottom')

plt.tight_layout()
plt.show()


# In[55]:


import matplotlib.pyplot as plt
import numpy as np

# Get the top 5 countries with the highest final values
top_countries = selected_data.drop("Year", axis=1).iloc[-1].sort_values(ascending=False).head(5).index

# Filter the data for top 5 countries
top_countries_df = selected_data[top_countries]

# Plotting
plt.figure(figsize=(10, 6))

# Set the positions for the bars
x = np.arange(len(selected_data["Year"]))

# Grouped bar plot for each country's mean values
bar_width = 0.15
for idx, country in enumerate(top_countries):
    bars = plt.bar(x + idx * bar_width, top_countries_df[country], width=bar_width, label=country)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, yval, round(yval, 2), va='bottom')

plt.xlabel('Year')
plt.ylabel('Mean Value Mt CO2')
plt.title('Year-wise Mean Values of Emission from Waste Sector for Top 5 Countries')
plt.xticks(x + bar_width * 2, selected_data["Year"].astype(int), rotation=45)
plt.legend(title='Countries')
plt.tight_layout()
plt.show()


# In[56]:


top_countries


# In[41]:


import matplotlib.pyplot as plt
import numpy as np

# Get the top 5 countries with the highest final values
top_countries = selected_data.drop("Year", axis=1).iloc[-1].sort_values(ascending=False).tail(5).index

# Filter the data for top 5 countries
top_countries_df = selected_data[top_countries]

# Plotting
plt.figure(figsize=(10, 6))

# Set the positions for the bars
x = np.arange(len(selected_data["Year"]))

# Grouped bar plot for each country's mean values
bar_width = 0.15
for idx, country in enumerate(top_countries):
    bars = plt.bar(x + idx * bar_width, top_countries_df[country].apply(lambda x: max(0, x)), width=bar_width, label=country)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, yval, round(yval, 2), va='bottom')

plt.xlabel('Year')
plt.ylabel('Mean Value Mt CO2')
plt.title('Year-wise Mean Values of Emission from Waste Sector for Bottom 5 Countries')
plt.xticks(x + bar_width * 2, selected_data["Year"].astype(int), rotation=45)
plt.legend(title='Countries')
plt.tight_layout()
plt.show() 


# In[57]:


# Filter data for the year 2030
data_2030 = forecast_df_arima_waste[forecast_df_arima_waste['Year'] == 2030]

# Calculate the sum of emissions for the top 5 countries
top_5_countries = ['Indonesia', 'United States', 'Russian Federation', 'Brazil', 'India'] # Replace with actual countries
sum_top_5 = data_2030[top_5_countries].sum().sum()

# Calculate the total emission from all countries
total_emission = data_2030.iloc[:, 1:].sum().sum()

# Calculate the percentage contribution of the top 5 countries
percentage_contribution = (sum_top_5 / total_emission) * 100

# Display the results
print("Sum of emissions for top 5 countries:", sum_top_5)
print("Total emission from all countries:", total_emission)
print("Percentage contribution of top 5 countries:", percentage_contribution)

# Create a pie chart
plt.figure(figsize=(6, 6))
plt.pie([percentage_contribution, 100 - percentage_contribution],
        labels=['Top 5 Countries', 'Other Countries'],
        autopct='%1.1f%%',
        colors=['#1f77b4', '#ff7f0e'],
        startangle=140)
plt.title("Percentage Contribution of Top 5 Countries to Total Emission - Waste sector (2030)")
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()


# In[34]:


# Extract the list of unique country names
country_names = df['Country Name'].unique()

# Set the ARIMA order (p, d, q)
p, d, q = 3, 1, 3  # Replace with your chosen values

# Dictionary to store forecasts for each country
forecasts_by_country = {}

# Loop through each country
for country in country_names:
    country_data = df[df['Country Name'] == country]['Agriculture']
    
    # Fit ARIMA model
    model = ARIMA(country_data, order=(p, d, q))
    model_fit = model.fit()
    
    # Forecast for the next 20 years
    forecast_steps = 20
    forecast = model_fit.forecast(steps=forecast_steps)
    
    # Store forecast in the dictionary
    forecasts_by_country[country] = forecast

# Convert the forecasts dictionary to a DataFrame
forecast_df_arima_agriculture = pd.DataFrame(forecasts_by_country)

# Add years to the DataFrame
years = range(df['Time'].max() + 1, df['Time'].max() + 1 + forecast_steps)  
forecast_df_arima_agriculture['Year'] = years 
forecast_df_arima_agriculture


# In[53]:


# Filter data for the year 2030
data_2030 = forecast_df_arima_agriculture[forecast_df_arima_agriculture['Year'] == 2030]

# Calculate the sum of emissions for the top 5 countries
top_5_countries = ['Brazil', 'Indonesia', 'Congo, Dem. Rep.', 'Canada', 'Myanmar'] # Replace with actual countries
sum_top_5 = data_2030[top_5_countries].sum().sum()

# Calculate the total emission from all countries
total_emission = data_2030.iloc[:, 1:].sum().sum()

# Calculate the percentage contribution of the top 5 countries
percentage_contribution = (sum_top_5 / total_emission) * 100

# Display the results
print("Sum of emissions for top 5 countries:", sum_top_5)
print("Total emission from all countries:", total_emission)
print("Percentage contribution of top 5 countries:", percentage_contribution)

# Create a pie chart
plt.figure(figsize=(6, 6))
plt.pie([percentage_contribution, 100 - percentage_contribution],
        labels=['Top 5 Countries', 'Other Countries'],
        autopct='%1.1f%%',
        colors=['#1f77b4', '#ff7f0e'],
        startangle=140)
plt.title("Percentage Contribution of Top 5 Countries to Total Emission - Agriculture sector (2030)")
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()


# In[48]:


# Filter the data for the specific years
selected_years = [2011, 2015, 2020, 2025, 2030]
selected_columns = ["Year"] + [country for country in forecast_df_arima_agriculture if country != "Year"]
selected_data = forecast_df_arima_agriculture[selected_columns][forecast_df_arima_agriculture["Year"].isin(selected_years)]

selected_data


# In[49]:


# Assuming forecast_df_arima_waste contains the data

# Calculate the mean across all countries' values for each year
forecast_df_arima_agriculture["Year"] = forecast_df_arima_agriculture["Year"].astype(pd.Int64Dtype())
df_mean = forecast_df_arima_agriculture.drop("Year", axis=1).mean(axis=1)

# Calculate percentage increase
percentage_increase = df_mean.pct_change() * 100

# Plotting
plt.figure(figsize=(10, 6))

# Plotting the line connecting the bar plots
plt.plot(forecast_df_arima_agriculture["Year"].astype(pd.Int64Dtype()), percentage_increase, marker='o', label='Percentage Increase', color='blue')

# Plotting the bar plot
bars = plt.bar(forecast_df_arima_agriculture["Year"].astype(pd.Int64Dtype()), percentage_increase, alpha=0.5)

plt.xlabel('Year')
plt.ylabel('Percentage Increase')
plt.title('Year-wise Percentage Increase in Emission from Agriculture Sector Across Countries')
ax = plt.gca()
ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
plt.xticks(rotation=45)
plt.grid(True)
plt.legend()

# Displaying percentage values on top of the bars
for bar, percentage in zip(bars, percentage_increase):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{percentage:.2f}%', ha='center', va='bottom')

plt.tight_layout()
plt.show()


# In[49]:


import matplotlib.pyplot as plt
import numpy as np

# Get the top 5 countries with the highest final values
top_countries = selected_data.drop("Year", axis=1).iloc[-1].sort_values(ascending=False).head(5).index

# Filter the data for top 5 countries
top_countries_df = selected_data[top_countries]

# Plotting
plt.figure(figsize=(10, 6))

# Set the positions for the bars
x = np.arange(len(selected_data["Year"]))

# Grouped bar plot for each country's mean values
bar_width = 0.15
for idx, country in enumerate(top_countries):
    bars = plt.bar(x + idx * bar_width, top_countries_df[country], width=bar_width, label=country)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, yval, round(yval, 2), va='bottom')

plt.xlabel('Year')
plt.ylabel('Mean Value Mt CO2')
plt.title('Year-wise Mean Values of Emission from Agriculture Sector for Top 5 Countries')
plt.xticks(x + bar_width * 2, selected_data["Year"].astype(int), rotation=45)
plt.legend(title='Countries')
plt.tight_layout()
plt.show()


# In[50]:


top_countries


# In[76]:


import matplotlib.pyplot as plt
import numpy as np

# Get the bottom 5 countries with the highest final values
bottom_countries = selected_data.drop("Year", axis=1).iloc[-1].sort_values(ascending=False).tail(106).index

# Filter the data for bottom 5 countries and values greater than 0
bottom_countries_df = selected_data[bottom_countries]
bottom_countries_df = bottom_countries_df.applymap(lambda x: max(0.01, x))

# Filter out similar values
bottom_countries_df = bottom_countries_df.loc[:, ~bottom_countries_df.T.duplicated(keep='first')].tail(5)

# Plotting
plt.figure(figsize=(10, 6))

# Set the positions for the bars
x = np.arange(len(selected_data["Year"]))

# Grouped bar plot for each country's mean values
bar_width = 0.15
for idx, country in enumerate(bottom_countries_df.columns):
    bars = plt.bar(x + idx * bar_width, bottom_countries_df[country], width=bar_width, label=country)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, yval, round(yval, 2), va='bottom')

plt.xlabel('Year')
plt.ylabel('Mean Value Mt CO2')
plt.title('Year-wise Mean Values of Emission from Agriculture Sector for Bottom 5 Countries')
plt.xticks(x + bar_width * 2, selected_data["Year"].astype(int), rotation=45)
plt.legend(title='Countries')
plt.tight_layout()
plt.show()


# In[23]:


import matplotlib.ticker as ticker
# Calculate the mean across all countries' values for each year
forecast_df_arima_agriculture["Year"] = forecast_df_arima_agriculture["Year"].astype(pd.Int64Dtype())
df_mean = forecast_df_arima_agriculture.drop("Year", axis=1).mean(axis=1)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(forecast_df_arima_agriculture["Year"].astype(pd.Int64Dtype()), df_mean, marker='o')
plt.xlabel('Year')
plt.ylabel('Mean Value Mt CO2 ')
plt.title('Year-wise Mean Values of Agriculture Emission Across Countries')
ax = plt.gca() 
ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()


# In[24]:


# Extract the list of unique country names
country_names = df['Country Name'].unique()

# Set the ARIMA order (p, d, q)
p, d, q = 2, 0, 2  
train_years = 2000  # Number of years for training
train_data = df[df['Time'] <= train_years]
test_data = df[df['Time'] > train_years]

# Dictionary to store forecasts and actual values for each country
forecasts_and_actuals_by_country = {}

# Loop through each country
for country in country_names:
    country_train_data = train_data[train_data['Country Name'] == country]['Agriculture']
    country_test_data = test_data[test_data['Country Name'] == country]['Agriculture']
    
    # Fit ARIMA model
    model = ARIMA(country_train_data, order=(p, d, q))
    model_fit = model.fit()
    
    # Forecast for the test set
    forecast = model_fit.forecast(steps=len(country_test_data))
    
    # Store forecast and actual values in the dictionary
    forecasts_and_actuals_by_country[country] = {'forecast': forecast, 'actual': country_test_data.values}

# Calculate Mean Squared Error (MSE) for each country
mse_by_country = {}
for country, values in forecasts_and_actuals_by_country.items():
    mse = mean_squared_error(values['actual'], values['forecast'])
    mse_by_country[country] = mse

# Convert the MSE dictionary to a DataFrame
mse_df = pd.DataFrame.from_dict(mse_by_country, orient='index', columns=['MSE'])
rmse
# Display the MSE for each country
print(mse_df)

# Calculate the overall average MSE
average_mse = math.sqrt(mse_df['MSE'].mean())
print(f"Average MSE across all countries: {average_mse:.2f}")


# In[35]:


average_mse = math.sqrt(mse_df['MSE'].mean())
print(f"Average RMSE across all countries: {average_mse:.2f}")


# In[ ]:


train_data


# In[59]:


# Extract the list of unique country names
country_names = df['Country Name'].unique()

# Set the ARIMA order (p, d, q)
p, d, q = 3, 1, 3  # Replace with your chosen values

# Dictionary to store forecasts for each country
forecasts_by_country = {}

# Loop through each country
for country in country_names:
    country_data = df[df['Country Name'] == country]['Energy'] 
    
    # Fit ARIMA model
    model = ARIMA(country_data, order=(p, d, q))
    model_fit = model.fit()
    
    # Forecast for the next 20 years
    forecast_steps = 20
    forecast = model_fit.forecast(steps=forecast_steps)
    
    # Store forecast in the dictionary
    forecasts_by_country[country] = forecast

# Convert the forecasts dictionary to a DataFrame
forecast_df_arima_energy = pd.DataFrame(forecasts_by_country)

# Add years to the DataFrame
years = range(df['Time'].max() + 1, df['Time'].max() + 1 + forecast_steps)  
forecast_df_arima_energy['Year'] = years 
forecast_df_arima_energy


# In[64]:


# Filter data for the year 2030
data_2030 = forecast_df_arima_energy[forecast_df_arima_energy['Year'] == 2030]

# Calculate the sum of emissions for the top 5 countries
top_5_countries = ['United States', 'India', 'Russian Federation', 'Japan', 'Korea, Rep.'] # Replace with actual countries
sum_top_5 = data_2030[top_5_countries].sum().sum()

# Calculate the total emission from all countries
total_emission = data_2030.iloc[:, 1:].sum().sum()

# Calculate the percentage contribution of the top 5 countries
percentage_contribution = (sum_top_5 / total_emission) * 100

# Display the results
print("Sum of emissions for top 5 countries:", sum_top_5)
print("Total emission from all countries:", total_emission)
print("Percentage contribution of top 5 countries:", percentage_contribution)

# Create a pie chart
plt.figure(figsize=(6, 6))
plt.pie([percentage_contribution, 100 - percentage_contribution],
        labels=['Top 5 Countries', 'Other Countries'],
        autopct='%1.1f%%',
        colors=['#1f77b4', '#ff7f0e'],
        startangle=140)
plt.title("Percentage Contribution of Top 5 Countries to Total Emission - Waste sector (2030)")
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()


# In[44]:


import matplotlib.pyplot as plt

# Sample data
labels = ['Energy Sector', 'Agriculture Sector', 'Waste Sector']
values = [452977.63585278153, 2140.770776317647, 1476.7202738187434]

# Create a horizontal bar chart
plt.figure(figsize=(10, 6))
plt.barh(labels, values, color='skyblue')

# Add labels and title
plt.xlabel('Values')
plt.ylabel('Sectors')
plt.title('Values by Sector')

# Display the chart
plt.show()


# In[38]:




# Calculate the mean across all countries' values for each year
forecast_df_arima_energy["Year"] = forecast_df_arima_energy["Year"].astype(pd.Int64Dtype())
df_mean = forecast_df_arima_energy.drop("Year", axis=1).mean(axis=1)

# Calculate percentage increase
percentage_increase = df_mean.pct_change() * 100

# Plotting
plt.figure(figsize=(10, 6))

# Plotting the line connecting the bar plots
plt.plot(forecast_df_arima_energy["Year"].astype(pd.Int64Dtype()), percentage_increase, marker='o', label='Percentage Increase', color='blue')

# Plotting the bar plot
bars = plt.bar(forecast_df_arima_energy["Year"].astype(pd.Int64Dtype()), percentage_increase, alpha=0.5)

plt.xlabel('Year')
plt.ylabel('Percentage Increase')
plt.title('Year-wise Percentage Increase in Emission from Energy Sector Across Countries')
ax = plt.gca()
ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
plt.xticks(rotation=45)
plt.grid(True)
plt.legend()

# Displaying percentage values on top of the bars
for bar, percentage in zip(bars, percentage_increase):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{percentage:.2f}%', ha='center', va='bottom')

plt.tight_layout()
plt.show()


# In[60]:


# Filter the data for the specific years
selected_years = [2011, 2015, 2020, 2025, 2030]
selected_columns = ["Year"] + [country for country in forecast_df_arima_energy if country != "Year"]
selected_data = forecast_df_arima_energy[selected_columns][forecast_df_arima_energy["Year"].isin(selected_years)]

selected_data['India']


# In[61]:


import matplotlib.pyplot as plt
import numpy as np

# Get the top 5 countries with the highest final values
top_countries = selected_data.drop("Year", axis=1).iloc[-1].sort_values(ascending=False).head(5).index

# Filter the data for top 5 countries
top_countries_df = selected_data[top_countries]

# Plotting
plt.figure(figsize=(10, 6))

# Set the positions for the bars
x = np.arange(len(selected_data["Year"]))

# Grouped bar plot for each country's mean values
bar_width = 0.15
for idx, country in enumerate(top_countries):
    bars = plt.bar(x + idx * bar_width, top_countries_df[country], width=bar_width, label=country)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, yval, round(yval, 2), va='bottom')

plt.xlabel('Year')
plt.ylabel('Mean Value Mt CO2')
plt.title('Year-wise Mean Values of Emission from Energy Sector for Top 5 Countries')
plt.xticks(x + bar_width * 2, selected_data["Year"].astype(int), rotation=45)
plt.legend(title='Countries')
plt.tight_layout()
plt.show()


# In[63]:


top_countries


# In[62]:


import matplotlib.pyplot as plt
import numpy as np

# Get the bottom 5 countries with the highest final values
bottom_countries = selected_data.drop("Year", axis=1).iloc[-1].sort_values(ascending=False).tail(5).index

# Filter the data for bottom 5 countries and values greater than 0
bottom_countries_df = selected_data[bottom_countries]
bottom_countries_df = bottom_countries_df.applymap(lambda x: max(0.01, x))

# Filter out similar values
bottom_countries_df = bottom_countries_df.loc[:, ~bottom_countries_df.T.duplicated(keep='first')]

# Plotting
plt.figure(figsize=(12, 6))

# Set the positions for the bars
x = np.arange(len(selected_data["Year"]))

# Grouped bar plot for each country's mean values
bar_width = 0.15
for idx, country in enumerate(bottom_countries_df.columns):
    bars = plt.bar(x + idx * bar_width, bottom_countries_df[country], width=bar_width, label=country)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, yval, round(yval, 2), va='center', ha='center', rotation='vertical')

plt.xlabel('Year')
plt.ylabel('Mean Value Mt CO2')
plt.title('Year-wise Mean Values of Emission from Energy Sector for Bottom 5 Countries')
plt.xticks(x + bar_width * 2, selected_data["Year"].astype(int), rotation=45)
plt.legend(title='Countries')
plt.tight_layout()
plt.show()


# In[132]:


import matplotlib.ticker as ticker
# Calculate the mean across all countries' values for each year
forecast_df_arima_energy["Year"] = forecast_df_arima_energy["Year"].astype(pd.Int64Dtype())
df_mean = forecast_df_arima_energy.drop("Year", axis=1).mean(axis=1)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(forecast_df_arima_energy["Year"].astype(pd.Int64Dtype()), df_mean, marker='o')
plt.xlabel('Year')
plt.ylabel('Mean Value MT CO2')
plt.title('Year-wise Mean Values of Emission from Energy Sector Across Countries')
ax = plt.gca()
ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()


# In[27]:


# Find the top 5 countries with the highest forecasted values
top_countries_arima_energy = forecast_df_arima_energy.drop("Year", axis=1).max().nlargest(5).index.tolist()

# Find the least 5 countries with the lowest forecasted values
least_countries_arima_energy  = forecast_df_arima_energy.drop("Year", axis=1).min().nsmallest(5).index.tolist()

print("Top 5 countries with highest forecasted values:")
print(top_countries_arima_energy)

print("\nLeast 5 countries with lowest forecasted values:")
print(least_countries_arima_energy)


# In[42]:


# Find the top 5 countries with the highest forecasted values
top_countries_arima_waste = forecast_df_arima_waste.drop("Year", axis=1).max().nlargest(5).index.tolist()

# Find the least 5 countries with the lowest forecasted values
least_countries_arima_waste  = forecast_df_arima_waste.drop("Year", axis=1).min().nsmallest(5).index.tolist()
  
print("Top 5 countries with highest forecasted values:") 
print(top_countries_arima_waste)

print("\nLeast 5 countries with lowest forecasted values:")
print(least_countries_arima_waste)


# In[100]:


# Find the top 5 countries with the highest forecasted values
top_countries_arima_agriculture = forecast_df_arima_agriculture.drop("Year", axis=1).max().nlargest(5).index.tolist()

# Find the least 5 countries with the lowest forecasted values
least_countries_arima_agriculture  = forecast_df_arima_agriculture.drop("Year", axis=1).min().nsmallest(5).index.tolist()
  
print("Top 5 countries with highest forecasted values:")    
print(top_countries_arima_agriculture)

print("\nLeast 5 countries with lowest forecasted values:")
print(least_countries_arima_agriculture)


# # SARIMA

# In[116]:


country_names = df['Country Name'].unique()  

# Define ranges for order and seasonal_order parameters
p_values = range(0, 3)  # Example range, adjust as needed
d_values = range(0, 2)  # Example range, adjust as needed
q_values = range(0, 3)  # Example range, adjust as needed
P_values = range(0, 2)  # Example range, adjust as needed
D_values = range(0, 2)  # Example range, adjust as needed
Q_values = range(0, 2)  # Example range, adjust as needed
s_values = [12]  # Seasonality period (e.g., 12 for monthly data)

# Create a dictionary to store forecasted values
forecast_dict = {}
order = (1, 1, 1)  # Replace with your chosen values
seasonal_order = (1, 1, 1, 12)  # Replace with your chosen values


# Perform country-wise forecasting
for country in country_names:
    country_data = df[df['Country Name'] == country]['Waste']


    # Fit the SARIMA model with the best parameters
    best_sarima_model = SARIMAX(country_data, order=order, seasonal_order=seasonal_order)
    best_sarima_fit = best_sarima_model.fit()

    # Forecast for the next 20 years
    forecast_steps = 20
    forecast = best_sarima_fit.get_forecast(steps=forecast_steps)

    # Store forecasted values in the dictionary
    forecast_dict[country] = forecast.predicted_mean

# Create a DataFrame from the forecast dictionary
forecast_df_sarima_waste = pd.DataFrame(forecast_dict)

# Display the forecasted DataFrame
forecast_df_sarima_waste  


# In[119]:


country_names = df['Country Name'].unique()  

# Define ranges for order and seasonal_order parameters
p_values = range(0, 3)  # Example range, adjust as needed
d_values = range(0, 2)  # Example range, adjust as needed
q_values = range(0, 3)  # Example range, adjust as needed
P_values = range(0, 2)  # Example range, adjust as needed
D_values = range(0, 2)  # Example range, adjust as needed
Q_values = range(0, 2)  # Example range, adjust as needed
s_values = [12]  # Seasonality period (e.g., 12 for monthly data)

# Create a dictionary to store forecasted values
forecast_dict = {}
order = (1, 1, 1)  # Replace with your chosen values
seasonal_order = (1, 1, 1, 12)  # Replace with your chosen values


# Perform country-wise forecasting
for country in country_names:
    country_data = df[df['Country Name'] == country]['Agriculture']


    # Fit the SARIMA model with the best parameters
    best_sarima_model = SARIMAX(country_data, order=order, seasonal_order=seasonal_order)
    best_sarima_fit = best_sarima_model.fit()

    # Forecast for the next 20 years
    forecast_steps = 20
    forecast = best_sarima_fit.get_forecast(steps=forecast_steps)

    # Store forecasted values in the dictionary
    forecast_dict[country] = forecast.predicted_mean

# Create a DataFrame from the forecast dictionary
forecast_df_sarima_agriculture = pd.DataFrame(forecast_dict)

# Display the forecasted DataFrame  
forecast_df_sarima_agriculture


# In[118]:


country_names = df['Country Name'].unique()  

# Define ranges for order and seasonal_order parameters
p_values = range(0, 3)  # Example range, adjust as needed
d_values = range(0, 2)  # Example range, adjust as needed
q_values = range(0, 3)  # Example range, adjust as needed
P_values = range(0, 2)  # Example range, adjust as needed
D_values = range(0, 2)  # Example range, adjust as needed
Q_values = range(0, 2)  # Example range, adjust as needed
s_values = [12]  # Seasonality period (e.g., 12 for monthly data)

# Create a dictionary to store forecasted values
forecast_dict = {}
order = (1, 1, 1)  # Replace with your chosen values
seasonal_order = (1, 1, 1, 12)  # Replace with your chosen values
rmse_by_country=[]

# Perform country-wise forecasting
for country in country_names:
    country_data = df[df['Country Name'] == country]['Energy']


    # Fit the SARIMA model with the best parameters
    best_sarima_model = SARIMAX(country_data, order=order, seasonal_order=seasonal_order)
    best_sarima_fit = best_sarima_model.fit()

    # Forecast for the next 20 years
    forecast_steps = 20
    forecast = best_sarima_fit.get_forecast(steps=forecast_steps)

    # Store forecasted values in the dictionary
    forecast_dict[country] = forecast.predicted_mean
    
      # Calculate RMSE for the forecast
    rmse = np.sqrt(mean_squared_error(country_data[-forecast_steps:], forecast.predicted_mean))


# Create a DataFrame from the forecast dictionary
forecast_df_sarima_energy  = pd.DataFrame(forecast_dict)

# Display the forecasted DataFrame
forecast_df_sarima_energy  


# In[ ]:


mean_by_year_f = forecast_df_sarima_energy.mean(axis=1)
mean_by_year_o=df.mean(axis=1)
# Display the mean values
print(mean_by_year)
mean_by_year_o


# In[ ]:


# Find the top 5 countries with the highest forecasted values
top_countries_sarima_waste = forecast_df_sarima_waste.max().nlargest(5).index.tolist()

# Find the least 5 countries with the lowest forecasted values
least_countries_sarima_waste  = forecast_df_sarima_waste.min().nsmallest(5).index.tolist()

print("Top 5 countries with highest forecasted values:")
print(top_countries_sarima_waste)  

print("\nLeast 5 countries with lowest forecasted values:")
print(least_countries_sarima_waste)


# In[ ]:


# Find the top 5 countries with the highest forecasted values
top_countries_sarima_agriculture = forecast_df_sarima_agriculture.max().nlargest(5).index.tolist()

# Find the least 5 countries with the lowest forecasted values
least_countries_sarima_agriculture  = forecast_df_sarima_agriculture.min().nsmallest(5).index.tolist()
  
print("Top 5 countries with highest forecasted values:")
print(top_countries_sarima_agriculture)   

print("\nLeast 5 countries with lowest forecasted values:")
print(least_countries_sarima_agriculture)


# In[ ]:


# Find the top 5 countries with the highest forecasted values
top_countries_sarima_energy = forecast_df_sarima_energy.max().nlargest(5).index.tolist()

# Find the least 5 countries with the lowest forecasted values
least_countries_sarima_energy  = forecast_df_sarima_energy.min().nsmallest(5).index.tolist()
  
print("Top 5 countries with highest forecasted values:")  
print(top_countries_sarima_energy)     

print("\nLeast 5 countries with lowest forecasted values:")
print(least_countries_sarima_energy)


# In[ ]:


def select_country(df, name, feature):
  country = df[df['Country Name'] == name]
  data = country[['Time', feature]]
  data['year'] = pd.date_range(start = '1990', end = '2011', freq='A')
  data = data.set_index('Time')
  return data

def decopose_df(df):
  decompose = seasonal_decompose(df)
  plt.figure(figsize=(20,12))
  decompose.plot()
  plt.show()
  return 

def plot_country_data(df, feature):
  return df.plot(xlabel = 'Date', ylabel = feature)

def split_data(df, train_size):
  train = df[: int(len(df)*train_size)]
  test = df[- math.ceil(len(df)*(1-train_size)) :]
  return train, test





def prophet(df, country, feature, train_size, fcast_num_years):
  data = df[df['country'] == country]
  fb_data = data[['year', feature]].rename(columns = {'year':'ds',feature:'y'})
  fb_data['ds'] = pd.date_range(start='1991',end='2021',freq='A')
  
  train = fb_data[: int(len(df)*train_size)]
  test = fb_data[- math.ceil(len(df)*(1-train_size)) :] 
 

  model = Prophet()
  model.fit(train)

  future = model.make_future_dataframe(
      periods=fcast_num_years, freq='A'
  )
  fcast = model.predict(future)

  model.plot(fcast);

  
  return fcast


# In[ ]:


Country_sector = select_country(df, 'United Kingdom', 'Waste')
#decopose_df(Country_sector)
Country_sector 


# In[ ]:


Country_sector['ds'] = pd.to_datetime(Country_sector['year'], format='%Y')
Country_sector.set_index('ds', inplace=True)

# Fit the ARIMA model
order = (1, 1, 1)  # Replace with the appropriate order for your data (p, d, q)
model = ARIMA(Country_sector['Waste'], order=order)
results = model.fit()

# Forecast for the next 10 years
forecast = results.forecast(steps=20)

# Add the forecasted values back to the DataFrame
future_dates = pd.date_range(start=Country_sector.index[-1], periods=20, freq='Y')[1:]  # Exclude the last date in the original data
forecast_df_arima_waste = pd.DataFrame({'Waste': forecast}, index=future_dates) 
#df = pd.concat([UK_co2, forecast_df_arima_energy]) 
 
plt.figure(figsize=(10, 6)) 
plt.plot(Country_sector.index, Country_sector['Waste'], label='Historical Data')
plt.plot(forecast_df_arima_waste.index, forecast_df_arima_waste['Waste'], label='Forecast', color='red')
plt.xlabel('Year')
plt.ylabel('Waste')
plt.title('ARIMA Forecast for the next 10 years')
plt.legend()
plt.show()


# In[ ]:


forecast_df_arima_waste


# In[ ]:


Country_sector = select_country(df, 'Australia', 'Agriculture')
Country_sector['ds'] = pd.to_datetime(Country_sector['year'], format='%Y')
Country_sector.set_index('ds', inplace=True)

# Fit the ARIMA model
order = (1, 1, 1)  # Replace with the appropriate order for your data (p, d, q)
model = ARIMA(Country_sector['Agriculture'], order=order)
results = model.fit()

# Forecast for the next 10 years
forecast = results.forecast(steps=20)

# Add the forecasted values back to the DataFrame
future_dates = pd.date_range(start=Country_sector.index[-1], periods=20, freq='Y')[1:]  # Exclude the last date in the original data
forecast_df_arima_waste = pd.DataFrame({'Agriculture': forecast}, index=future_dates) 
#df = pd.concat([UK_co2, forecast_df_arima_energy]) 
 
plt.figure(figsize=(10, 6)) 
plt.plot(Country_sector.index, Country_sector['Agriculture'], label='Historical Data')
plt.plot(forecast_df_arima_waste.index, forecast_df_arima_waste['Agriculture'], label='Forecast', color='red')
plt.xlabel('Year')
plt.ylabel('Agriculture')
plt.title('ARIMA Forecast for the next 10 years')
plt.legend()
plt.show()


# In[ ]:


forecast_df_arima_waste


# In[ ]:


window = 3
df['Moving_Average'] = df['Energy'].rolling(window=window).mean()

# Print the DataFrame with moving average forecasts
print(df)


# In[ ]:


UK_co2.columns


# In[ ]:


# Prepare the DataFrame for Simple Exponential Smoothing
UK_co2['ds'] = pd.to_datetime(UK_co2['year'], format='%Y-%m-%d')
UK_co2.set_index('ds', inplace=True)

# Fit the Simple Exponential Smoothing model
alpha = 0.5  # Smoothing parameter (0 < alpha < 1)
model = SimpleExpSmoothing(UK_co2['CC.GHG.EMSE'])
fit = model.fit(smoothing_level=alpha)

# Forecast for the next 10 years
last_year = UK_co2.index[-1].year
future_years = list(range(last_year + 1, last_year + 11))
forecast = fit.forecast(steps=10)

# Add the forecasted values back to the DataFrame
future_dates = pd.date_range(start=UK_co2.index[-1], periods=11, freq='Y')[1:]  # Exclude the last date in the original data
forecast_df = pd.DataFrame({'CC.GHG.EMSE': forecast}, index=future_dates)
df = pd.concat([UK_co2, forecast_df])

# Plot the forecasted values
plt.figure(figsize=(10, 6))
plt.plot(UK_co2.index, UK_co2['CC.GHG.EMSE'], label='Historical Data')
plt.plot(forecast_df.index, forecast_df['CC.GHG.EMSE'], label='Forecast', color='red')
plt.xlabel('Year')
plt.ylabel('CC.GHG.EMSE')
plt.title('Simple Exponential Smoothing Forecast for the next 10 years')
plt.legend()
plt.show()


# In[ ]:


# Fit the Holt-Winters Linear Model
alpha = 0.5  # Smoothing parameter for the level (0 < alpha < 1)
beta = 0.5   # Smoothing parameter for the trend (0 < beta < 1)
model = ExponentialSmoothing(UK_co2['CC.GHG.EMSE'], trend='add', seasonal=None)
fit = model.fit(smoothing_level=alpha, smoothing_slope=beta)

# Forecast for the next 10 years
last_year = UK_co2.index[-1].year
future_years = list(range(last_year + 1, last_year + 11))
forecast = fit.forecast(steps=20)

# Add the forecasted values back to the DataFrame
future_dates = pd.date_range(start=UK_co2.index[-1], periods=11, freq='Y')[1:]  # Exclude the last date in the original data
forecast_df = pd.DataFrame({'CC.GHG.EMSE': forecast}, index=future_dates)
df = pd.concat([UK_co2, forecast_df])

# Plot the forecasted values
plt.figure(figsize=(10, 6))
plt.plot(UK_co2.index, UK_co2['CC.GHG.EMSE'], label='Historical Data')
plt.plot(forecast_df.index, forecast_df['CC.GHG.EMSE'], label='Forecast', color='red')
plt.xlabel('Year')
plt.ylabel('CC.GHG.EMSE')
plt.title('Holt-Winters Linear Model Forecast for the next 10 years')
plt.legend()
plt.show()


# In[ ]:


last_year


# In[ ]:


# Fit the SARIMA model
order = (1, 1, 1)  # Order of non-seasonal part (p, d, q)
seasonal_order = (1, 1, 1, 12)  # Seasonal order (P, D, Q, seasonal_periods)
model = SARIMAX(UK_co2['Waste'], order=order, seasonal_order=seasonal_order)
fit = model.fit()

# Forecast for the next 10 years
last_year = UK_co2.index[-1]
future_years = pd.date_range(start=last_year + pd.DateOffset(years=1), periods=20, freq='A')
forecast = fit.get_forecast(steps=20)

# Get confidence intervals for the forecast
forecast_conf_int = forecast.conf_int()

# Add the forecasted values back to the DataFrame 
forecast_df = pd.DataFrame({'Energy': forecast.predicted_mean,
                            'lower_conf': forecast_conf_int['lower Energy'],
                            'upper_conf': forecast_conf_int['upper Energy']},
                           index=future_years)
#df = pd.concat([UK_co2, forecast_df])

# Plot the forecasted values with confidence intervals
plt.figure(figsize=(10, 6))
plt.plot(UK_co2.index, UK_co2['Energy'], label='Historical Data')
plt.plot(forecast_df.index, forecast_df['Energy'], label='Forecast', color='red')
plt.fill_between(forecast_df.index, forecast_df['lower_conf'], forecast_df['upper_conf'], color='gray', alpha=0.3)
plt.xlabel('Year')
plt.ylabel('Energy')
plt.title('SARIMA Forecast for the next 10 years')
plt.legend()
plt.show()


# In[ ]:


decomposition = seasonal_decompose(UK_co2['CC.GHG.EMSE'], model='additive', period=10)  # Assuming data is seasonal with a period of 12 (yearly)

# Plot the decomposed components
plt.figure(figsize=(12, 8))
plt.subplot(4, 1, 1)
plt.plot(UK_co2.index, UK_co2['CC.GHG.EMSE'], label='Original Data')
plt.legend(loc='upper left')
plt.subplot(4, 1, 2)
plt.plot(UK_co2.index, decomposition.trend, label='Trend')
plt.legend(loc='upper left')
plt.subplot(4, 1, 3)
plt.plot(UK_co2.index, decomposition.seasonal, label='Seasonal')
plt.legend(loc='upper left')
plt.subplot(4, 1, 4)
plt.plot(UK_co2.index, decomposition.resid, label='Residual')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()


# In[ ]:


fig = go.Figure()

# Add the actual values as a scatter trace
fig.add_trace(go.Scatter(x=UK_co2.index, y=UK_co2['CC.GHG.EMSE'], mode='lines+markers', name='Actual'))

# Add the forecasted values as a scatter trace
fig.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df['CC.GHG.EMSE'], mode='lines+markers', name='Forecast'))

# Update layout
fig.update_layout(title='ARIMA Forecast for the next 10 years', xaxis_title='Year', yaxis_title='CC.GHG.EMSE')

# Show the plot
fig.show()


# In[ ]:


data = UK_co2.reset_index()
data = data.rename(columns={'ds': 'ds', 'CC.GHG.EMSE': 'y'})

# Create and fit the Prophet model
model = Prophet()
model.fit(data)

# Forecast for the next 20 years
future = model.make_future_dataframe(periods=20, freq='Y')
forecast = model.predict(future)

# Plot the forecasted values
fig, ax = plt.subplots(figsize=(10, 6))
model.plot(forecast, ax=ax)
ax.plot(data['ds'], data['y'], 'k.', label='Historical Data')
ax.set_xlabel('Year')
ax.set_ylabel('CC.GHG.EMSE')
ax.set_title('Prophet Time Series Forecast for the next 20 years')
plt.legend()
plt.show()


# In[ ]:




