# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 10:25:37 2020

@author: Bhaskar """

#import libraries
import  numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing dataset
dataset = pd.read_csv('data.csv')

X=dataset.iloc[:, 2:32].values
y=dataset.iloc[:,1].values

#Encoding Categorical data
from sklearn.preprocessing import LabelEncoder
labelencode_y= LabelEncoder()
y = labelencode_y.fit_transform(y)

#splitting the dataset into train set and test set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size = 0.1, random_state =0)

# feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)

# fitting the data in random forest model
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 250,random_state=0 )
regressor.fit(X_train,y_train)

#predicting results usuing the modal
y_pred = regressor.predict(X_test)

y_pred=np.rint(y_pred)


#creating the confusion matrix
from sklearn.metrics import confusion_matrix
cm1 =confusion_matrix(y_test,y_pred)

#Fitting the model in KNN model
from sklearn.neighbors import KNeighborsClassifier
classifier =  KNeighborsClassifier(n_neighbors = 8,metric = 'minkowski',p=2)
classifier.fit(X_train,y_train)

#predicting results from KNN
y1_pred=classifier.predict(X_test)

#creating the confusion matrix
from sklearn.metrics import confusion_matrix
cm2=confusion_matrix(y_test,y1_pred)



