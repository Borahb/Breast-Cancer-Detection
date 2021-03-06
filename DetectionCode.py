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


#calculating f1 score
from sklearn.metrics import f1_score
f1_score(y_test,y1_pred)

#calculating accuracy score

from sklearn.metrics import accuracy_score
accuracy_score(y_test,y1_pred)

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

#calculating f1 score
from sklearn.metrics import f1_score
f1_score(y_test,y1_pred)

#calculating accuracy score

from sklearn.metrics import accuracy_score
accuracy_score(y_test,y1_pred)

#fitting the data in Naive Bayes model
from sklearn.naive_bayes import GaussianNB
classi = GaussianNB()
classi.fit(X_train,y_train)

#predicting reults
y2_pred = classi.predict(X_test)

#creating the confusion matrix
from sklearn.metrics import confusion_matrix
cm3=confusion_matrix(y_test,y2_pred)

#calculating f1 score
from sklearn.metrics import f1_score
f1_score(y_test,y2_pred)

#calculating accuracy score

from sklearn.metrics import accuracy_score
accuracy_score(y_test,y2_pred)


#now fitting the data in decision tree classification
from sklearn.tree import DecisionTreeClassifier
classi2 = DecisionTreeClassifier(criterion = "entropy",random_state =0)
classi2.fit(X_train,y_train)

#predicting results
y3_pred = classi2.predict(X_test)


#creating the confusion matrix
from sklearn.metrics import confusion_matrix
cm4=confusion_matrix(y_test,y3_pred)

#calculating f1 score
from sklearn.metrics import f1_score
f1_score(y_test,y3_pred)

#calculating accuracy score

from sklearn.metrics import accuracy_score
accuracy_score(y_test,y3_pred)

#fitting the data in random forst classification model
from sklearn.ensemble import RandomForestClassifier
classifi2 = RandomForestClassifier(n_estimators = 180,criterion = "entropy",random_state =0)
classifi2.fit(X_train,y_train)

#predicting results
y4_pred = classifi2.predict(X_test)

#creating the confusion matrix
from sklearn.metrics import confusion_matrix
cm5=confusion_matrix(y_test,y4_pred)

#calculating f1 score
from sklearn.metrics import f1_score
f1_score(y_test,y4_pred)

#calculating accuracy score

from sklearn.metrics import accuracy_score
accuracy_score(y_test,y4_pred)

## Pickle
import pickle
 
# save model
pickle.dump(classifi2, open('breast_cancer_detector.pickle', 'wb'))
 
# load model
breast_cancer_detector_model = pickle.load(open('breast_cancer_detector.pickle', 'rb'))
 
# predict the output
yf_pred = breast_cancer_detector_model.predict(X_test)
 
# confusion matrix
print('Confusion matrix of Random Forest Classifier: \n',confusion_matrix(y_test, y_pred),'\n')
 
# show the accuracy
print('Accuracy of Random Forest Classifier = ',accuracy_score(y_test, y_pred))

