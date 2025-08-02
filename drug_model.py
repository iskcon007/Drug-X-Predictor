#Important libraries 
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
#pandas, numpy: For data manipulation.
# joblib: To save the trained model to disk.
# train_test_split: To split the dataset into training and testing sets.
# LabelEncoder: Converts categorical (text) data into numbers.
# RandomForestClassifier: The machine learning model used.
# metrics: To evaluate the model's performance.


#Load the dataset
file_path = "Drug Consumption (Test Dataset).csv"
df = pd.read_csv(file_path)
#Loads the dataset from a CSV file into a pandas DataFrame.

#Initial Exploration
print("First 5 rows:\n", df.head())
print("\nNull values:\n", df.isnull().sum())
print("\nDuplicates:", df.duplicated().sum())
print("Shape:", df.shape)
# df.head(): Displays the first 5 rows.
# df.isnull().sum(): Checks for any missing values in each column.
# df.duplicated().sum(): Checks for duplicate rows.
# df.shape: Shows how many rows and columns the dataset has.


#Target Variable Engineering
df['Cannabis_Binary'] = df['Cannabis'].apply(lambda x: 0 if x in ['CL0', 'CL1', 'CL2'] else 1)
# Creates a binary label:
# 0: Non-user / low usage (CL0, CL1, CL2)
# 1: Moderate to heavy user (other classes)

#Drop Unnecessary Columns
df = df.drop(['Gender', 'Country', 'Cannabis'], axis=1)
# These columns are removed:
# Gender and Country (possibly irrelevant or already encoded)
# Cannabis (replaced by Cannabis_Binary)

#Feature and Label Split
X = df.drop('Cannabis_Binary', axis=1)
y = df['Cannabis_Binary']
# X: All input features (predictors)
# y: Target variable (Cannabis_Binary)

#Encode Categorical Features
for col in X.columns:
    if X[col].dtype == 'object':
        X[col] = LabelEncoder().fit_transform(X[col])
#Converts any text/categorical features to numerical using Label Encoding.

#Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Splits the data into:
# 80% Training
# 20% Testing
# random_state=42 ensures reproducibility.

#Model Training
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
#Trains a Random Forest classifier on the training data.

#Make Predictions
y_pred = model.predict(X_test)
#Uses the trained model to make predictions on the test data.

#Evaluate the Model
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Accuracy Score:", accuracy_score(y_test, y_pred))
# Confusion Matrix: Shows TP, TN, FP, FN.
# Classification Report: Precision, Recall, F1-score.
# Accuracy Score: Overall % of correct predictions.

#Save the Model
joblib.dump(model, "cannabis_rf_model.joblib")
print("\nModel saved as cannabis_rf_model.joblib")
#Saves the trained model as a .joblib file so it can be reused without retraining.