import pandas as pd
import numpy as np

#Machine Learning Algos
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression

#Used to split dataset into Training and testing data
from sklearn.model_selection import train_test_split

#Used to evaluate Classification models
from sklearn.metrics import accuracy_score, confusion_matrix

#convert text columns into numerical values
from sklearn.preprocessing import LabelEncoder

#Part A: Dataset understanding
#   q1
df= pd.read_csv("Dataset 2.csv")
print(df.head())

#Q2: 
print(df.shape)

#3:
print(df.columns)

#4:
# Numerical columns
numerical_features = df.select_dtypes(exclude=['object']).columns

# Categorical columns
categorical_features = df.select_dtypes(include=['object']).columns

print("Numerical Features:",numerical_features)

print("\nCategorical Features:",categorical_features)

#5:

print(df.isnull().sum().any())
#Indicates no missing value

#Exploratory Data Analysis

#6:
print("The average age of users:")
print(df['Age'].mean())

#7: 
print("The average watch hours per week:")
print(df['WatchHoursPerWeek'].mean())

#8:
print("the avg monthly spending of users:")
print(df['MonthlySpend'].mean())

#9:
print(df.groupby('SubscriptionType').size()) 

#10:
renewed_count = (df['SubscriptionRenewed'] == 'Yes').sum()
total_users = len(df)

renewed_percentage = (renewed_count / total_users) * 100
print("The percentage of users who renewed their subscription are:")
print(renewed_percentage)

#11:
le= LabelEncoder()
df['Gender']= le.fit_transform(df['Gender'])
df['SubscriptionType']= le.fit_transform(df['SubscriptionType'])
df['SubscriptionRenewed']=le.fit_transform(df['SubscriptionRenewed'])
df['FavoriteGenre']= le.fit_transform(df['FavoriteGenre'])

#12:
X = df.drop(['UserID', 'SubscriptionRenewed'], axis=1)

y = df['SubscriptionRenewed']

#13:
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

#14:
dt= DecisionTreeClassifier(random_state=42)
dt.fit (X_train, y_train)
pred= dt.predict(X_test)

#15:
print('Decision Tree Accuarcy:', accuracy_score(y_test,pred))

#16:
print("/nConfusion matrix")
print(confusion_matrix(y_test,pred))

#17:
from sklearn.neighbors import KNeighborsClassifier

knn_model = KNeighborsClassifier(n_neighbors=5)

knn_model.fit(X_train, y_train)

#18:
knn_pred = knn_model.predict(X_test)
knn_accuracy = accuracy_score(y_test, knn_pred)

print(knn_accuracy)

#Compare the Accuracy:
print("Decision Tree:", accuracy_score(y_test,pred))

print("KNN:", knn_accuracy)

#19: Linear Regression
X_reg = df.drop(['UserID', 'MonthlySpend'], axis=1)

y_reg = df['MonthlySpend']
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg,
    y_reg,
    test_size=0.2,
    random_state=42
)
lr_model = LinearRegression()

lr_model.fit(X_train_reg, y_train_reg)

#20:
new_user = [[25, 1, 2, 30, 3, 0, 15, 1]]
prediction = lr_model.predict(new_user)

print("Predicted Monthly Spending:", prediction[0])


#Business Questions:
'''
Q1. The factors that seem to influence subscription renewal are:

WatchHoursPerWeek – users who watch more content are more likely to renew.
SubscriptionType – Premium/VIP users may show higher renewal rates.
MonthlySpend – users who spend more tend to remain subscribed.
DevicesUsed – users using multiple devices may be more engaged with the platform.

Q2. 
Subscription Renewal is a certainly a classification problem as it allows
categories into either Yes Or nO. 

Q3.
Monthly Spending includes: numbers n has continuous data , hence is a regression
problem.

Q4.
Visibly, KNN has better accuracy than decision tree classifier, hence KNN can have said to be 
performed better.

Q5.
Netflix can use these predictions to:

Identify users who are likely to cancel their subscriptions.
Provide personalized recommendations.
Offer discounts or special renewal offers.
Send reminders before subscription expiry.
Recommend suitable subscription plans based on user behavior.

'''
