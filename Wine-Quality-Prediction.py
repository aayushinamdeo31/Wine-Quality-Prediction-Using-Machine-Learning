#!/usr/bin/env python
# coding: utf-8

# # Wine-Quality-Prediction System

# ### Steps involved -
# 1. Collection of data
# 2. Data Preprocessing
# 3. Model Building
# 4. Testing
# 5. Prediction
# 6. Accuracy
# 7. Confusion Matrix
# 8. ROC and AUC curves

# ### Importing Necessary Libraries

# In[ ]:


import numpy as np #used for numerical analysis
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,confusion_matrix

import sklearn.metrics as metrics


# In[ ]:


# load data


# In[ ]:


dataset = pd.read_csv("C:\\Users\\Aayushii\\Desktop\\smart_bridge\\wine_quality\\winequalityN.csv")


# In[ ]:


dataset.head(5)


# In[ ]:


# Data Preprocessing


# In[ ]:


dataset.describe()


# In[ ]:


dataset.info()


# In[ ]:


dataset.isnull().any() # checking for any null values


# In[ ]:


dataset.isnull().sum() # counting all null values


# In[ ]:


# Filling null values with mean, median, mode

dataset['fixed acidity'].fillna(dataset['fixed acidity'].mean(), inplace = True)
dataset['volatile acidity'].fillna(dataset['volatile acidity'].mean(), inplace = True)
dataset['citric acid'].fillna(dataset['citric acid'].mean(), inplace = True)
dataset['residual sugar'].fillna(dataset['residual sugar'].mean(), inplace = True)
dataset['chlorides'].fillna(dataset['chlorides'].mean(), inplace = True)
dataset['pH'].fillna(dataset['pH'].mean(), inplace = True)
dataset['sulphates'].fillna(dataset['sulphates'].mean(), inplace = True)


# In[ ]:


# Data Visualization


# In[ ]:


plt.scatter(x="quality",y="alcohol", data=dataset)
plt.title("Wine Quality Prediction")
plt.xlabel("Quality")
plt.ylabel("Alcohol")
plt.plot()


# In[ ]:


dataset['type'].unique()


# In[ ]:


quality_mapping = { 3 : "Low",4 : "Low",5: "Medium",6 : "Medium",7: "Medium",8 : "High",9 : "High"}
dataset["quality"] =  dataset["quality"].map(quality_mapping)


# In[ ]:


mapping_quality = {"Low" : 0,"Medium": 1,"High" : 2}
dataset["quality"] =  dataset["quality"].map(mapping_quality)
dataset.head(5)


# In[ ]:


le = LabelEncoder()
dataset['type'] = le.fit_transform(dataset['type'])
dataset.head(5)


# In[ ]:


x = dataset.iloc[:, 0:12].values
y = dataset.iloc[:, 12:13].values
x.shape


# In[ ]:


x_train,x_test,y_train,y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)


# In[ ]:


sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)


# In[ ]:


rfc = RandomForestClassifier(n_estimators = 12, criterion = "entropy", random_state = 0)
rfc.fit(x_train, y_train)


# In[ ]:

import pickle
pickle.dump(rfc, open('C:\\Users\\Aayushii\\Desktop\\smart_bridge\\wine_quality\\wine_quality_model.pkl','wb'))
rfc_pred = rfc.predict(x_test)
rfc_pred


# In[ ]:


rfc_acc = accuracy_score(y_test, rfc_pred)
rfc_acc


# In[ ]:


rfc_cm = confusion_matrix(y_test, rfc_pred)
rfc_cm


# rfc_fpr, rfc_tpr, threshold = metrics.roc_curve(y_test, rfc_pred)
# rfc_roc_auc = metrics.auc(rfc_fpr, rfc_tpr)

# In[ ]:


rfc_p = rfc.predict(sc.transform([[1, 7.0, 0.27, 0.36, 20.7, 0.045, 45.0, 170.0, 1, 3, 0.45 , 8.8]]))
rfc_p

