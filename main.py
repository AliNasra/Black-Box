import pandas as pd
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from filter_aflite import filter
import numpy as np

# test on the available csv files under the data directory
# It's assumed that problem is a classification one

def train_filtered(add,encode):
    # Load the dataset
    address = add
    data    = filter(address)
    target  = None
    if data.columns.tolist()[-1] != "id":
        target  = data.columns.tolist()[-1]
    else:
        target  = data.columns.tolist()[-2]
    # Encode the target variable
    if encode:
        label_encoder = LabelEncoder()
        data[target]  = label_encoder.fit_transform(data[target])
    # Split the data
    col_to_drop   = [target,"id"]
    X = data.drop(col_to_drop, axis=1)
    y = data[target]
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    # Calculate and print the average accuracy
    average_accuracy = np.mean(scores)
    print(f'Accuracy Filtered: {average_accuracy:.2f}')

def train_unfiltered(add,encode):
    # Load the dataset
    data    = pd.read_csv(add)
    target  = data.columns.tolist()[-1]
    # Encode the target variable
    if encode:
        label_encoder = LabelEncoder()
        data[target]  = label_encoder.fit_transform(data[target])
    # Split the data
    X = data.drop(target, axis=1)
    y = data[target]
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    # Calculate and print the average accuracy
    average_accuracy = np.mean(scores)
    print(f'Accuracy Unfiltered: {average_accuracy:.2f}')

encode  = True # True if the target column isn't encoded numerically. False if it is
address = 'data/rice_cammeo.csv'
train_unfiltered(address,encode)
train_filtered(address,encode)

