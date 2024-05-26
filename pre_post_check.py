import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from filter_aflite import filter


def no_filtering(f1,f2):
    # Load your dataset
    data_male        = pd.read_csv(f1)
    data_female      = pd.read_csv(f2)
    last_column_name = data_female.columns[-1]



    # Split the data into features and target
    X_male = data_male.drop(last_column_name, axis=1)  # Replace 'target_column' with the name of your target column
    y_male = data_male[last_column_name]
    X_female = data_male.drop(last_column_name, axis=1)  # Replace 'target_column' with the name of your target column
    y_female = data_male[last_column_name]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_male, y_male, test_size=0.2, random_state=42)

    # Train the Random Forest model
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # Evaluate the model
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Before Filtering")
    print(f'Accuracy Male Prediction: {accuracy}')
    y_pred = clf.predict(X_female)
    accuracy = accuracy_score(y_female, y_pred)
    print(f'Accuracy Female Prediction: {accuracy}')


def filtering(f1,f2):
    # Load your dataset
    data_male        = filter(f1)
    data_female      = pd.read_csv(f2)
    last_column_name = data_female.columns[-1]

    # Split the data into features and target
    X_male = data_male.drop(last_column_name, axis=1)  # Replace 'target_column' with the name of your target column
    y_male = data_male[last_column_name]
    X_female = data_male.drop(last_column_name, axis=1)  # Replace 'target_column' with the name of your target column
    y_female = data_male[last_column_name]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_male, y_male, test_size=0.2, random_state=42)

    # Train the Random Forest model
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # Evaluate the model
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("After Filtering")
    print(f'Accuracy Male Prediction: {accuracy}')
    y_pred = clf.predict(X_female)
    accuracy = accuracy_score(y_female, y_pred)
    print(f'Accuracy Female Prediction: {accuracy}')


f1            = "Cardio/males.csv"
f2            = "Cardio/females.csv"
filter_choice = True
if filter_choice:
    filtering(f1,f2)
else:
    no_filtering(f1,f2)