# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 16:32:00 2019

@author: Shyam
"""

# Load libraries
import matplotlib.pyplot as plt
#setting dimension of graph
plt.rcParams["figure.figsize"]= (12, 7)

import pandas as pd
import numpy as np
import seaborn as sns

from pandas.plotting import scatter_matrix

from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score,roc_auc_score





# Load dataset
path = "C:/Users/shara/Desktop/NNDL/bank-full.csv"
dataset = pd.read_csv(path,delimiter=';')


#Information of data
dataset.info()

#check for any missing values
dataset.apply(lambda x: sum(x.isnull()),axis=0)



# Target variable distribution
count = dataset.groupby('y').size()
percent = count/len(dataset)*100
print(percent)


# scatter plot matrix
scatter_matrix(dataset)
plt.show()

plt.matshow(dataset.corr())
plt.show()

corr = dataset.corr()
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)
#####################################################################################################################
# Impute outliers function
def impute_outliers(df, column , minimum, maximum):
    col_values = df[column].values
    df[column] = np.where(np.logical_or(col_values<minimum, col_values>maximum), col_values.mean(), col_values)
    return df

# age
sns.boxplot(x='y', y='age', data=dataset)

#balance
sns.boxplot(x='y', y='balance', data=dataset)


# Fixing balance column
dataset_new = dataset
min_val = dataset_new["balance"].min()
max_val = 20000
dataset_new = impute_outliers(df=dataset_new, column='balance' , minimum=min_val, maximum=max_val)

#day
sns.boxplot(x='y', y='day', data=dataset)

#duration
sns.boxplot(x='y', y='duration', data=dataset)

# Fixing duration column
min_val = dataset_new["duration"].min()
max_val = 2000
dataset_new = impute_outliers(df=dataset_new, column='duration' , minimum=min_val, maximum=max_val)


sns.boxplot(x='y', y='campaign', data=dataset)


# Fixing campaign column
min_val = dataset_new["campaign"].min()
max_val = 20
dataset_new = impute_outliers(df=dataset_new, column='campaign' , minimum=min_val, maximum=max_val)

#pdays
sns.boxplot(x='y', y='pdays', data=dataset)

# Fixing pdays column
min_val = dataset_new["pdays"].min()
max_val = 250
dataset_new = impute_outliers(df=dataset_new, column='pdays' , minimum=min_val, maximum=max_val)

#previos
sns.boxplot(x='y', y='previous', data=dataset)


# Fixing 'previous' column
min_val = dataset_new["previous"].min()
max_val = 15
dataset_new = impute_outliers(df=dataset_new, column='previous' , minimum=min_val, maximum=max_val)




#lets see statistic of Numerical variables after Outlier treatment
dataset_new.describe()

#####################################################################################################################

# Impute unknowns function
def impute_unknowns(df, column):
    col_values = df[column].values
    df[column] = np.where(col_values=='unknown', dataset[column].mode(), col_values)
    return df


# job
temp1 = pd.crosstab(dataset['job'], dataset['y'])
temp1.plot(kind='bar')
print(dataset.groupby(['job']).size()/len(dataset)*100)


# Fixing 'job' column
dataset_new = impute_unknowns(df=dataset_new, column='job')

# marital
temp2 = pd.crosstab(dataset['marital'], dataset['y'])
temp2.plot(kind='bar')
print(dataset.groupby(['marital']).size()/len(dataset)*100)

# education
temp3 = pd.crosstab(dataset['education'], dataset['y'])
temp3.plot(kind='bar')
print(dataset.groupby(['education']).size()/len(dataset)*100)

# Fixing 'education' column
dataset_new = impute_unknowns(df=dataset_new, column='education')

# default
temp4 = pd.crosstab(dataset['default'], dataset['y'])
temp4.plot(kind='bar')
print(dataset.groupby(['default']).size()/len(dataset)*100)

#dropping variable 'default'
del dataset_new['default']

# housing
temp5 = pd.crosstab(dataset['housing'], dataset['y'])
temp5.plot(kind='bar')
print(dataset.groupby(['housing']).size()/len(dataset)*100)

# "contact"
temp6 = pd.crosstab(dataset['contact'], dataset['y'])
temp6.plot(kind='bar')
print(dataset.groupby(['contact']).size()/len(dataset)*100)

del dataset_new['contact']

# "month"
temp7 = pd.crosstab(dataset['month'], dataset['y'])
temp7.plot(kind='bar')
print(dataset.groupby(['month']).size()/len(dataset)*100)

# "poutcome"
temp8 = pd.crosstab(dataset['poutcome'], dataset['y'])
temp8.plot(kind='bar')
print(dataset.groupby(['poutcome']).size()/len(dataset)*100)


#dropping variable 'poutcome'
del dataset_new['poutcome']


# "loan"
temp9 = pd.crosstab(dataset['loan'], dataset['y'])
temp9.plot(kind='bar')
print(dataset.groupby(['loan']).size()/len(dataset)*100)


#####################################################################################################################

dataset_Y = dataset_new['y']
dataset_X = dataset_new[dataset_new.columns[0:13]]


dataset_X_dummy = pd.get_dummies(dataset_X)
print(dataset_X_dummy.head())


#converting dataframe into numpy Array
X = dataset_X_dummy.values
Y = dataset_Y.values

# Split-out validation dataset
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)

#Scaling the values
X_t = scale(X_train)

#initially lets create 40 components which is actual number of Variables we have
pca = PCA(n_components=40)

pca.fit(X_t)

#The amount of variance that each PC explains
var= pca.explained_variance_ratio_

#Cumulative Variance explains
var1=np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)

#lets see Cumulative Variance plot
plt.plot(var1)



#Looking at above plot I'm taking 33 variables
pca = PCA(n_components=33)
pca.fit(X_t)
X_train_PC=pca.fit_transform(X_t)

    
#Scaling the X_validation data
X_v = scale(X_validation)

pca.fit(X_v)
X_validation_PC=pca.fit_transform(X_v)


#####################################################################################################################

#####################################################################################################################


from sklearn import model_selection
from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

le = preprocessing.LabelEncoder()

seed=42
scoring = 'accuracy'


   

sm = SMOTE()
X_train_PC_s, Y_train_s = sm.fit_sample(X_train_PC, Y_train)


   
nb=GaussianNB()
nb.fit(X_train_PC_s, Y_train_s)
prediction=nb.predict(X_validation_PC)
kfold = model_selection.KFold(n_splits=10, random_state=seed)
cv_results = model_selection.cross_val_score(nb, X_train_PC_s, Y_train_s, cv=kfold, scoring=scoring)
a=accuracy_score(Y_validation, prediction)
confusionmatrix = confusion_matrix(Y_validation, prediction)
cr=classification_report(Y_validation, prediction)

le.fit(Y_validation)
list(le.classes_)
y_=le.transform(Y_validation) 
p_=le.transform(prediction)
area=roc_auc_score(y_,p_)


print('Training Accuracy')
print(cv_results.mean())
print('Testing Accuracy')
print(a)
print('Confusion Matrix')    
print(confusionmatrix)    
print('Report')    
print(cr)   
print('AUC ROC Score')    
print(area)    

