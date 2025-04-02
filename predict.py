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
print(X_encoded)
print(y)

X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Model Accuracy: {accuracy:.2f}")


# Define a new passenger's details
new_passenger = pd.DataFrame({
    'Pclass': [3],      # Third class
    'Sex': ['male'],     # Male
    'Embarked': ['S']    # Embarked at Southampton
})

# Apply the same encoding transformation
new_passenger_encoded = encoder.transform(new_passenger)

# Convert to DataFrame to match training format
new_passenger_encoded = pd.DataFrame(new_passenger_encoded, columns=encoder.get_feature_names_out(categorical_features))

# Make a prediction
prediction = model.predict(new_passenger_encoded)
prediction_proba = model.predict_proba(new_passenger_encoded)

# Display the result
print(f"Predicted Survival: {'Survived' if prediction[0] == 1 else 'Did Not Survive'}")
print(f"Prediction Probability: {prediction_proba[0]}")
