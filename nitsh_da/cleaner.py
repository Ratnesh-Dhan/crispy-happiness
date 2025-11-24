import pandas as pd
import os

# Load the dataset
data = pd.read_csv('/mnt/d/DATASETS/titanic.csv')

# Display the first few rows of the dataset
print(data.head())

# Dropping unnecessary columns
data = data.drop(columns=['Name', 'Ticket', 'Cabin'])

# Handle missing values
# Fill missing Age values with the median
data['Age'].fillna(data['Age'].median(), inplace=True)

# Fill missing Embarked values with the most frequent value (mode)
data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)

# Drop rows with missing Fare (if any)
data = data.dropna(subset=['Fare'])

# Convert categorical variables using one-hot encoding
data = pd.get_dummies(data, columns=['Sex', 'Embarked'], drop_first=True)

# Normalize continuous variables (Age, Fare)
data['Age'] = (data['Age'] - data['Age'].mean()) / data['Age'].std()
data['Fare'] = (data['Fare'] - data['Fare'].mean()) / data['Fare'].std()

os.makedirs('../dataset', exist_ok=True)
data.to_csv('../dataset/titanic.csv', index=False)
# Display the cleaned dataset
print(data.info())
