---
layout: post
toc: true
title: "Titanic Survival Prediction with Logistic Regression"
categories: data-science
tags: [machine learning, pandas, sklearn, titanic]
author:
  - Luke Hutchings
date: 2025-10-08
---

# import library
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# Create DataFrame
data = pd.read_csv('train.csv')

# Display the DataFrame
data.head()

# creating a backup copy of the data 
data_original = data.copy()

# Populating null Age values with the average age by sex, Pclass, and Survived
data['Age'] = data.groupby(['Sex', 'Pclass'],group_keys=False)['Age'].apply(lambda x: x.fillna(x.mean()))

# Plot before and after imputation
fig, axes = plt.subplots(1, 2, figsize=(6, 8), sharey=True)
sns.histplot(data_original['Age'], ax=axes[0], kde=True, color='red').set_title('Before Imputation')
sns.histplot(data['Age'], kde=True, ax=axes[1], color='green').set_title('After Imputation')
plt.show()

# Convert categorical variables into dummy/indicator variables
data = pd.get_dummies(data, columns=['Sex', 'Embarked'], drop_first=True)

data.head()

#add your answer here

# Define features we will use for the model
X = data[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex_male', 'Embarked_Q', 'Embarked_S']]

# define the target variable 
y = data['Survived']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Train the logistic regression model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model using accuracy and confusion matrix
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print(f'Confusion Matrix:\n{conf_matrix}')

# Calculate feature importance
feature_importance = model.coef_[0]

# Create a DataFrame to display feature importance
importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': feature_importance
}).sort_values(by='Importance', ascending=False)

# Plot feature importance
plt.figure(figsize=(6, 4))
sns.barplot(x='Importance', y='Feature', data=importance_df)
plt.title('Feature Importance')
plt.show()

# Import new test data
test_data = pd.read_csv('test.csv')

# Populating null Age values with the average age by sex, Pclass, and Survived
test_data['Age'] = test_data.groupby(['Sex', 'Pclass'],group_keys=False)['Age'].apply(lambda x: x.fillna(x.mean()))

# check for null values 
test_data.isnull().sum()

# using an average of sex and PClass for the missing fare value
test_data['Fare'] = test_data.groupby(['Sex', 'Pclass'],group_keys=False)['Fare'].apply(lambda x: x.fillna(x.mean()))

# Preprocess the test data in the same way as the training data
test_data = pd.get_dummies(test_data, columns=['Sex', 'Embarked'], drop_first=True)

# Ensure the test data has the same columns as the training data
test_data = test_data.reindex(columns=X.columns, fill_value=0)

# Predict on the new test data
test_predictions = model.predict(test_data)

# adding the survived field back to the test data
test_data['Survived_predicated'] = test_predictions

# add your answer below
data['FamilySize'] = data['SibSp'] + data['Parch']

X = data[['Pclass', 'Age', 'FamilySize', 'Fare', 'Sex_male', 'Embarked_Q', 'Embarked_S']]

Y = data[['Survived']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=6)

model = LogisticReagression(max_iter=200)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy1 = accuracy_score(y_test, y_pred)
conf_matrix1 = confusion_matrix(y_test, y_pred)

print(f'Accuracy: {accuracy1}')
print(f'Confusion Matrix:\n{conf_matrix1}')

feature_importance = model.coef_[0]

#DataFrame to display feature importance
importance_df1 = pd.DataFrame({
    'Feature': X.columns,
    'Importance': feature_importance
}).sort_values(by='Importance', ascending=False)

# Plot feature importance
plt.figure(figsize=(6, 4))
sns.barplot(x='Importance', y='Feature', data=importance_df1)
plt.title('Feature Importance of Survived')
plt.show()


