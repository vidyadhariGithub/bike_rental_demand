import os
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn import tree
from sklearn import metrics
from math import sqrt
from sklearn.metrics import mean_squared_log_error

def normalize(data,columns):
    for i in columns:
        minimum , maximum = data[i].min(), data[i].max()
        data[i] = (data[i] - minimum)/(maximum - minimum)
    return data

training_dataset = pd.read_csv('train.csv', header = 0)
testing_dataset = pd.read_csv('test.csv', header = 0)

def gethours(dateCol):
    L = dateCol.split()
    hour = L[1].split(':')
    return int(hour[0])

def getYear(dateCol):
    L = dateCol.split()
    year = L[0].split('-')
    return int(year[0])

def getMonth(dateCol):
    L = dateCol.split()
    month = L[0].split('-')
    return int(month[1])

def getDay(dateCol):
    L = dateCol.split()
    day = L[0].split('-')
    return int(day[2])
    
def assign_label(hour):
    if hour > 6 and hour <= 12:
        return 1
    elif hour > 12 and hour <= 18:
        return 2
    elif hour > 18 and hour <= 24:
        return 3
    else:
        return 4

training_dataset['hr'] = training_dataset['datetime'].apply(gethours)
training_dataset['time_label'] = training_dataset['hr'].apply(assign_label)

training_dataset['year'] = training_dataset['datetime'].apply(getYear)
training_dataset['month'] = training_dataset['datetime'].apply(getMonth)
training_dataset['day'] = training_dataset['datetime'].apply(getDay)

testing_dataset['hr'] = testing_dataset['datetime'].apply(gethours)
testing_dataset['time_label'] = testing_dataset['hr'].apply(assign_label)
testing_dataset['year'] = testing_dataset['datetime'].apply(getYear)
testing_dataset['month'] = testing_dataset['datetime'].apply(getMonth)
testing_dataset['day'] = testing_dataset['datetime'].apply(getDay)

columns = training_dataset.columns.drop(['count', 'casual', 'datetime', 'registered', 'temp'])
#Org_data = Org_data.drop(columns, axis = 1)

testing_dataset = testing_dataset.drop(['datetime'],axis=1)

columns_to_normalize = ['temp', 'atemp', 'windspeed', 'humidity']
training_dataset = normalize(training_dataset, columns_to_normalize)
testing_dataset = normalize(testing_dataset, columns_to_normalize)
#training_dataset[columns].head(10)

Numerical_col = ['temp', 'atemp', 'humidity', 'windspeed', 'casual', 'registered', 'count']

#Outliers
for i in Numerical_col:
    last75, first25 = np.percentile(training_dataset.loc[:,i],[75,25])
    val=last75-first25
    a, b = round(first25-(val*1.5),4),round(last75+(val*1.5),4)
    outliers = len([n for n in training_dataset[i] if n > b or n < a])
    if outliers != 0: 
        print ('Outliers found in feature ',i)
        plt.figure(figsize=(12,0.8))
        sns.boxplot(training_dataset[i], color = 'g')
        plt.show()
    else: print('No outliers found',i)
        
Outliers_deleted_data = training_dataset.copy()
for i in ['humidity','windspeed', 'casual', 'registered', 'count']:
    q75, q25 = np.percentile(Outliers_deleted_data.loc[:,i], [75 ,25])
    iqr = q75 - q25
    min, max = q25 - (iqr*1.5),q75 + (iqr*1.5)
    Outliers_deleted_data = Outliers_deleted_data.drop(Outliers_deleted_data[Outliers_deleted_data.loc[:,i] < min].index)
    Outliers_deleted_data = Outliers_deleted_data.drop(Outliers_deleted_data[Outliers_deleted_data.loc[:,i] > max].index)
    
def dataloss(data1,data2):
    print('Data loss : ')
    print('Features deleted: ', data1.shape[1]-data2.shape[1])
    print('Samples deleted: ',data1.shape[0]-data2.shape[0])

dataloss(training_dataset,Outliers_deleted_data)


#DATA SPLITTING
x_train,x_test,y_train,y_test = train_test_split(training_dataset[columns], training_dataset['count'], 
                                                 test_size = 0.2, random_state = 0)

import math

hundred_percent_values = math.floor(training_dataset.shape[0]*1)
train_100 = training_dataset.sample(n=hundred_percent_values, random_state = 1)

hundred_percent_values = math.floor(testing_dataset.shape[0]*1)
test_100 = testing_dataset.sample(n=hundred_percent_values, random_state = 1)

x2_train,x2_test,y2_train,y2_test = train_test_split(Outliers_deleted_data[columns],Outliers_deleted_data['count'], 
                                                 test_size = 0.2, random_state = 0)

def reg_acc(y_true, y_pre):
    return_var = []
    rmse = sqrt(mean_squared_error(y_true,y_pre))
    return_var.append(rmse)
    print ("RMSE : ",rmse )
    r2 = r2_score(y_true,y_pre)
    return_var.append(r2)
    print ("R² : ",r2 )
    mae = mean_absolute_error(y_true,y_pre)
    return_var.append(mae)
    print ('MAE:',mae)
    rmsle =np.sqrt(mean_squared_log_error( y_true, y_pre ))
    return_var.append(rmsle)
    print('RMSLE:',rmsle)
    if 0 in y_true : 
        print("MAPE can't be calculated")
        return_var.append(0)
    else :
        mape = round(np.mean(np.abs((y_true - y_pre)/y_true))*100,4)
        print ('MAPE :', mape)
        return_var.append(mape)
        return_var.append(100-mape)
    return return_var

#outliers removed
knn_model = KNeighborsRegressor(n_neighbors= 2).fit(x2_train,y2_train)
knn_estimated = knn_model.predict(x2_test)
print("Error scores :\n")
knn_result = reg_acc(y2_test,knn_estimated)
mse = mean_squared_error(y2_test, knn_estimated)
print("Mean squared error :", mse)

#plot graph
train_counts = y2_train.values
train_counts_min = train_counts.min()
train_counts_max = train_counts.max()

test_counts = knn_estimated
test_counts_min, test_counts_max = test_counts.min(), test_counts.max()

plt.scatter(x2_train['hr'] , train_counts)
plt.scatter(x2_test['hr'] , test_counts, color='g')
plt.xlabel("Hours") 
plt.ylabel("Counts") 
plt.title("Counts at every hour of 24 months") 
plt.show()

#outliers present
knn_model = KNeighborsRegressor(n_neighbors= 2).fit(x_train, y_train)
knn_estimated = knn_model.predict(x_test)
mse = mean_squared_error(y_test, knn_estimated)
print("Error scores :\n")
knn_result = reg_acc(y_test,knn_estimated)
print("Mean squared error :", mse)

#plot graph
train_counts = y_train.values
train_counts_min = train_counts.min()
train_counts_max = train_counts.max()

test_counts = knn_estimated
test_counts_min, test_counts_max = test_counts.min(), test_counts.max()

plt.scatter(x_train['hr'] , train_counts)
plt.scatter(x_test['hr'] , test_counts, color='g')
plt.xlabel("Hours") 
plt.ylabel("Counts") 
plt.title("Counts at every hour of 24 months") 
plt.show()

#outliers present but 100% training data
knn_model = KNeighborsRegressor(n_neighbors= 2).fit(train_100[columns], train_100['count'])
knn_estimated = knn_model.predict(test_100[columns])

#plot graph
train_counts = train_100['count']
train_counts_min = train_counts.min()
train_counts_max = train_counts.max()

test_counts = knn_estimated
test_counts_min, test_counts_max = test_counts.min(), test_counts.max()

plt.scatter(train_100['hr'] , train_counts)
plt.scatter(test_100['hr'] , test_counts, color='g')
plt.xlabel("Hours") 
plt.ylabel("Counts") 
plt.title("Counts at every hour of 24 months") 
plt.show()

#Cross-Validation
from yellowbrick.model_selection import CVScores
from sklearn.model_selection import KFold

X_train, X_test, y_train, y_test = train_test_split(train_100[columns], train_100['count'], test_size=0.20)

dt = KNeighborsRegressor(n_neighbors= 2)
dt_fit = dt.fit(X_train, y_train)

cv = KFold(n_splits=12, random_state=42)

visualizer = CVScores(dt, cv=cv, scoring='r2')

visualizer.fit(train_100[columns], train_100['count'])
visualizer.show()