#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn.model_selection as ms
import numpy.random as nr
from sklearn import feature_selection as fs
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error,  r2_score
from sklearn.linear_model import Ridge


# In[2]:


df=pd.read_excel('D:\\Dissertation AM\\CCDR.xlsx') 
df = df.iloc[:-5]
df


# In[3]:


all_vars_clean = df

#define an array with the unique year values
years_count_missing = dict.fromkeys(all_vars_clean['Time'].unique(), 0)
for ind, row in all_vars_clean.iterrows():
    years_count_missing[row['Time']] += row.isnull().sum()

# sort the years by missing values
years_missing_sorted = dict(sorted(years_count_missing.items(), key=lambda item: item[1]))

# print the missing values for each year
print("missing values by year:")
for key, val in years_missing_sorted.items():
    print(key, ":", val)


# In[5]:


# Plotting
plt.bar(years_missing_sorted.keys(), years_missing_sorted.values())
plt.xlabel("Year")
plt.ylabel("Missing Values")
plt.title("Missing Values by Year")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[4]:


# Specify the year frame you want to extract
start_year = 1992
end_year = 2018

# Filter the DataFrame to include data within the specified year frame
filtered_df = df[(df['Time'] >= start_year) & (df['Time'] <= end_year)]

# Display the filtered DataFrame
filtered_df


# In[5]:


# check the amount of missing values by country
all_vars_clean=filtered_df
# define an array with the unique country values
countries_count_missing = dict.fromkeys(all_vars_clean['Country Name'].unique(), 0)

# iterate through all rows and count the amount of NaN values for each country
for ind, row in all_vars_clean.iterrows():
    countries_count_missing[row['Country Name']] += row.isnull().sum()

# sort the countries by missing values
countries_missing_sorted = dict(sorted(countries_count_missing.items(), key=lambda item: item[1]))

# print the missing values for each country
print("missing values by country:")
for key, val in countries_missing_sorted.items():
    print(key, ":", val)


# In[6]:


print("number of missing values in the whole dataset before filtering the countries:")
print(all_vars_clean.isnull().sum().sum())
print("number of rows before filtering the countries:")
print(all_vars_clean.shape[0])


# filter only rows for countries with less than 90 missing values
countries_filter = []
for key, val in countries_missing_sorted.items():
    if val<80:
        countries_filter.append(key)

all_vars_clean = all_vars_clean[all_vars_clean['Country Name'].isin(countries_filter)]

print("number of missing values in the whole dataset after filtering the countries:")
print(all_vars_clean.isnull().sum().sum())
print("number of rows after filtering the countries:")
print(all_vars_clean.shape[0])


# In[7]:


all_vars_clean.isnull().sum()


# In[8]:


from itertools import compress

# create a boolean mapping of features with more than 20 missing values
vars_bad = all_vars_clean.isnull().sum()>200

# remove the columns corresponding to the mapping of the features with many missing values
all_vars_clean2 = all_vars_clean.drop(compress(data = all_vars_clean.columns, selectors = vars_bad), axis='columns')

print("Remaining missing values per column:")
print(all_vars_clean2.isnull().sum())


# In[9]:


# delete rows with any number of missing values
all_vars_clean3 = all_vars_clean2.dropna(axis='rows', how='any')

print("Remaining missing values per column:")
print(all_vars_clean3.isnull().sum())

print("Final shape of the cleaned dataset:")
print(all_vars_clean3.shape)


# In[ ]:





# In[10]:


data=all_vars_clean3

# select all features
features_all = data

# plot a correlation of all features
# correlation matrix
sns.set(font_scale=2)
f,ax=plt.subplots(figsize=(30,20))
sns.heatmap(features_all.corr(), annot=True, cmap='coolwarm', fmt = ".2f", center=0, vmin=-1, vmax=1)
plt.title('Correlation between features', fontsize=25, weight='bold' )
plt.show()

sns.set(font_scale=1)


# In[11]:


# List of columns to drop
columns_to_drop = ['Electricity_consumption', 'Electricity_generation']  # Replace with the actual column names

# Drop the specified columns
data = data.drop(columns=columns_to_drop)
# select all features
features_all = data

# plot a correlation of all features
# correlation matrix
sns.set(font_scale=2)
f,ax=plt.subplots(figsize=(30,20))
sns.heatmap(features_all.corr(), annot=True, cmap='coolwarm', fmt = ".2f", center=0, vmin=-1, vmax=1)
plt.title('Correlation between features', fontsize=25, weight='bold' )
plt.show()

sns.set(font_scale=1)


# In[12]:


X = data.drop(['Per_capita_GHG_emissions','Country Name'], axis = 1)
y = data['Per_capita_GHG_emissions']
y


# In[13]:


feature_cols = ['Time', 'Agricultural_land', 'Access_to_electricity', 'GDP',
       'GDP_per_capita', 'Percapitadaily_coalconsumption',
       'Percapitadaily_gasconsumption', 'Percapitadaily_oilconsumption',
       'Population', 'Urban_population']


# In[14]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state=42) 


# In[17]:


# Set folds for cross validation for the feature selection
nr.seed(1)
feature_folds = ms.KFold(n_splits=4, shuffle = True, random_state=42)

# Define the model
rf_selector = RandomForestRegressor(random_state=42) 

# Define an objects for a model for recursive feature elimination with CV
nr.seed(1)
selector = fs.RFECV(estimator = rf_selector, cv = feature_folds, scoring = 'r2', n_jobs=-1)

selector = selector.fit(X_train, np.ravel(y_train))
selector.support_

print("Feature ranking after RFECV:")
print(selector.ranking_)

# print the important features
ranks_transform = list(np.transpose(selector.ranking_))
chosen_features = [i for i,j in zip(feature_cols,ranks_transform) if j==1]
print("Chosen important features:")
print(chosen_features)


# In[18]:


selected_columns = ['Agricultural_land', 'Access_to_electricity', 'GDP', 'GDP_per_capita', 'Percapitadaily_coalconsumption', 'Percapitadaily_gasconsumption', 'Percapitadaily_oilconsumption', 'Population', 'Urban_population']
X = data[selected_columns]  


# In[19]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state=42)


# In[20]:


# Initialize the Random Forest model
random_forest = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model on the training data
random_forest.fit(X_train, y_train)

# Make predictions on the testing data
predictions = random_forest.predict(X_test)

# Calculate R-squared
r2 = r2_score(y_test, predictions)
print("R-squared:", r2)

# Calculate Root Mean Squared Error (RMSE)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
print("Root Mean Squared Error:", rmse)


# In[21]:


# Calculate the number of accurate predictions
accurate_predictions = sum(abs(predictions - y_test))

# Calculate accuracy based on the number of accurate predictions
accuracy = accurate_predictions / len(y_test)
print("Accuracy:", accuracy)


# In[29]:


# Initialize the XGBoost model for regression
xgb_model = XGBRegressor(n_estimators=100, random_state=42)  # You can adjust hyperparameters

# Train the model on the training data
xgb_model.fit(X_train, y_train)

# Make predictions on the testing data
predictions = xgb_model.predict(X_test)

# Calculate R-squared
r2 = r2_score(y_test, predictions)
print("R-squared:", r2)

# Calculate Root Mean Squared Error (RMSE)
rmse = mean_squared_error(y_test, predictions, squared=False)
print("Root Mean Squared Error:", rmse)


# In[30]:


accurate_predictions = sum(abs(predictions - y_test))

# Calculate accuracy based on the number of accurate predictions
accuracy = accurate_predictions / len(y_test)
print("Accuracy:", accuracy)


# In[ ]:





# In[ ]:




