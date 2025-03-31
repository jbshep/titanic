import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

df = pd.read_csv('titanic.csv')

print(df.head())

categorical_features = ['Pclass', 'Sex', 'Embarked']
target = 'Survived'

# Drop people rows who have NA's for a categorical feature
df = df[categorical_features + [target]].dropna()

X = df[categorical_features]
y = df[target]
