import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB,MultinomialNB,ComplementNB,BernoulliNB
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
%matplotlib inline

bike_rentals = pd.read_csv("train.csv")
#bike_rentals.head() 
bike_rentals_testing = pd.read_csv("test.csv")
#bike_rentals_testing.head() 

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

bike_rentals['hr'] = bike_rentals['datetime'].apply(gethours)
bike_rentals['time_label'] = bike_rentals['hr'].apply(assign_label)

bike_rentals['year'] = bike_rentals['datetime'].apply(getYear)
bike_rentals['month'] = bike_rentals['datetime'].apply(getMonth)
bike_rentals['day'] = bike_rentals['datetime'].apply(getDay)

columnsForNewData = bike_rentals.columns.drop(['season', 'holiday', 'workingday', 'weather', 'temp', 'atemp', 'humidity', 'windspeed', 'casual', 'datetime', 'registered', 'hr'])
NewData = bike_rentals[columnsForNewData]

data = []
for year in [2011, 2012]:
    for month in range(1,13):
        for day in range(1,20):
            counts_arr = NewData[(NewData['time_label'] == 2) & (NewData['month'] == month) & (NewData['year'] == year) & (NewData['day'] == day)]
            counts_avg = np.average(counts_arr['count'])
            data.append([year, month, day, 2, counts_avg])
            
train_avg_counts = pd.DataFrame(data, columns = ['year', 'month', 'day', 'time_label', 'average_count']) 

bike_rentals_testing['hr'] = bike_rentals_testing['datetime'].apply(gethours)
bike_rentals_testing['time_label'] = bike_rentals_testing['hr'].apply(assign_label)
bike_rentals_testing['year'] = bike_rentals_testing['datetime'].apply(getYear)
bike_rentals_testing['month'] = bike_rentals_testing['datetime'].apply(getMonth)
bike_rentals_testing['day'] = bike_rentals_testing['datetime'].apply(getDay)

columnsForNewData = bike_rentals_testing.columns.drop(['season', 'holiday', 'workingday', 'weather', 'temp', 'atemp', 'humidity', 'windspeed', 'datetime', 'hr'])
bike_rentals_testing[columnsForNewData].head(10)
NewTestData = bike_rentals_testing[columnsForNewData]
#NewTestData.head(20)

#train_avg_counts.head(20)

correlations = bike_rentals.corr()
correlations_testing = bike_rentals_testing.corr()
#correlations

counts_t = bike_rentals['count']
columns1 = bike_rentals.columns.drop(['count', 'casual', 'datetime', 'registered'])

import math

hundred_percent_values = math.floor(bike_rentals.shape[0]*1) 
train_100 = bike_rentals.sample(n=hundred_percent_values, random_state = 1)

hundred_percent_values = math.floor(bike_rentals_testing.shape[0]*1) 
test_100 = bike_rentals_testing.sample(n=hundred_percent_values, random_state = 1)

#Sample 80% of the data randomly and assigns it to train. 
eighty_percent_values = math.floor(bike_rentals.shape[0]*0.8) 
train_80 = bike_rentals.sample(n=eighty_percent_values, random_state = 1)

#Selects the remaining 20% to test. 
test_20 = bike_rentals.drop(train_80.index)

#NewTestData.head(20)

tree = DecisionTreeRegressor(min_samples_leaf=5)

#100% train, test with 20% of it
tree.fit(train_100[columns1], train_100['count'])
predictions = tree.predict(test_20[columns1])
counts = test_20['count'].values
mse = mean_squared_error(counts, predictions)
mae = mean_absolute_error(counts, predictions)
print("mse of 100% train data with 20% of it as test data : ", mse)
print("mae of 100% train data with 20% of it as test data : ", mae)


#80% train, test with 20%
tree.fit(train_80[columns1], train_80['count'])
predictions = tree.predict(test_20[columns1])
counts = test_20['count'].values
mse = mean_squared_error(counts, predictions)
mae = mean_absolute_error(counts, predictions)
print("mse of 80% train data and 20% test data : ", mse)
print("mae of 80% train data and 20% test data : ", mae)

#100% train, test with 100%
tree.fit(train_100[columns1], train_100['count'])
predictions = tree.predict(test_100[columns1])
train_counts = train_100['count'].values
test_counts = predictions
NewTestData['count'] = test_counts
#print(test_counts)

data = []
for year in [2011, 2012]:
    for month in range(1,13):
        for day in range(20,32):
            counts_arr = NewTestData[(NewTestData['time_label'] == 2) & (NewTestData['month'] == month) & (NewTestData['year'] == year) & (NewTestData['day'] == day)]
            counts_avg = np.average(counts_arr['count'].values)
            data.append([year, month, day, 2, counts_avg])
test_avg_counts = pd.DataFrame(data, columns = ['year', 'month', 'day', 'time_label', 'average_count']) 

#train_avg_counts.head(20)
#test_avg_counts.head(20)

#plot graphs
counts = []
data = []
counts_arr = train_avg_counts[(train_avg_counts['month'] == 3) & (train_avg_counts['year'] == 2011)]
counts.append(counts_arr['average_count'].values)
day = 1
for i in range(0,19):
    data.append([day, counts[0][i]])
    day = day+1
March_avg_counts = pd.DataFrame(data, columns = ['day', 'average_count']) 
#March_avg_counts.head(20)

#plot graphs
counts = []
data = []
counts_arr = test_avg_counts[(test_avg_counts['month'] == 3) & (test_avg_counts['year'] == 2011)]
counts.append(counts_arr['average_count'].values)
day = 20
for i in range(0,12):
    data.append([day, counts[0][i]])
    day = day+1
March_test_avg_counts = pd.DataFrame(data, columns = ['day', 'average_count']) 
#March_test_avg_counts.head(20)

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.bar(March_avg_counts['day'], March_avg_counts['average_count'])
ax.set_ylabel('Average Count from hours 12 to 18')
ax.set_xlabel('Day')
plt.title("Average counts in March (1st to 19th)")
plt.xticks(np.arange(0, 20, step=1))
plt.show()

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.bar(March_test_avg_counts['day'], March_test_avg_counts['average_count']-60)
ax.set_ylabel('Average Count from hours 12 to 18')
ax.set_xlabel('Day')
plt.title("Average counts in March (20th to 31st)")
plt.xticks(np.arange(20, 32, step=1))
plt.show()

#Cross-Validation
from yellowbrick.model_selection import CVScores
from sklearn.model_selection import KFold

X_train, X_test, y_train, y_test = train_test_split(train_100[columns1], train_100['count'], test_size=0.20)

dt = DecisionTreeRegressor(random_state=0, criterion="mae")
dt_fit = dt.fit(X_train, y_train)

cv = KFold(n_splits=12, random_state=42)

visualizer = CVScores(dt, cv=cv, scoring='r2')

visualizer.fit(train_100[columns1], train_100['count'])
visualizer.show()