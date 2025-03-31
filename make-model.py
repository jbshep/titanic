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

encoder = OneHotEncoder(drop='first', sparse_output=False)
X_encoded = encoder.fit_transform(X)
X_encoded = pd.DataFrame(X_encoded, columns=encoder.get_feature_names_out(categorical_features))

X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)
