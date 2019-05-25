# -*- coding: utf-8 -*-
"""
Created on Wed May 15 11:03:38 2019

@author: piyush
"""
'''Perform the following on the credit.xlsx dataset (1+3+3+3+3+3+4) 

a. Read dataset and drop 5 columns - Loan_ID, Dependents, Gender, Self_Employed, Married column. Consider Loan_Status as class label for classification. 
b. Diagrammatically show if the data consists of any null values. If yes then preprocess by replacing null values with mean value.
c. Display correlation of independent variables and display the two variables which are highly correlated 
d. Process categorical variables for X and y both 
e. Split the data into 80:20 ratio i.e. 80% train data and 20% test data and calculate accuracy after applying logistic regression on it i.e. train the model and test it. (Note : Apply Feature Scaling before classification)
f. Display confusion matrix and classification report.
g. Display ROC curve.
'''
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

data = pd.read_excel('credit_B.xlsx')
data.drop(['Loan_ID', 'Dependents', 'Gender', 'Self_Employed', 'Married'], axis=1, inplace=True)

sns.heatmap(data.isnull(),yticklabels=False,cbar=False,cmap='viridis')

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer = imputer.fit(data['LoanAmount'].values.reshape(-1, 1))
data['LoanAmount'] = imputer.transform(data['LoanAmount'].values.reshape(-1, 1))
imputer = imputer.fit(data['Loan_Amount_Term'].values.reshape(-1, 1))
data['Loan_Amount_Term'] = imputer.transform(data['Loan_Amount_Term'].values.reshape(-1, 1))
imputer = imputer.fit(data['Credit_History'].values.reshape(-1, 1))
data['Credit_History'] = imputer.transform(data['Credit_History'].values.reshape(-1, 1))

multicolinearity_check = data.corr()

print("LoanAmount and ApplicantIncome are highly correlated")

X = data.iloc[:, :-1] #Take all the columns except last one
y = data.iloc[:, -1] #Take the last column as the result

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X.iloc[:, 0] = labelencoder_X.fit_transform(X.iloc[:, 0])
X.iloc[:, 6] = labelencoder_X.fit_transform(X.iloc[:, 6])

#Make dummy variables
onehotencoder = OneHotEncoder(categorical_features = [6])
X = onehotencoder.fit_transform(X).toarray()

# Avoiding the Dummy Variable Trap
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,random_state=101)

# Feature Scaling #Need to be done after splitting
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train.iloc[:, [0,2]] = sc.fit_transform(X_train.iloc[:, [0,2]])
X_test.iloc[:, [0,2]] = sc.transform(X_test.iloc[:, [0,2]])

# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train, y_train)




#Find relevant features
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV

# The "accuracy" scoring is proportional to the number of correct
# classifications
rfecv = RFECV(estimator=classifier, step=1, cv=StratifiedKFold(2), scoring='accuracy')
rfecv.fit(X_train, y_train)

print("Optimal number of features : %d" % rfecv.n_features_)

# K-Fold cross validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=10)
model_accuracy = accuracies.mean()
model_standard_deviation = accuracies.std()

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True)

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
