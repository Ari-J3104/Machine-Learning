#adult

import pandas as pd
from sklearn import preprocessing 
#import MaxAbsScaler, MinMaxScaler, Normalizer, RobustScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report



df = pd.read_csv('AdultDatabase.csv')

# Convert text values to numerical format
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = pd.factorize(df[col])[0]

# Separate the features and labels
X = df.iloc[:,:-1]
y = df.iloc[:,-1]




scalers = [preprocessing.MaxAbsScaler(), preprocessing.MinMaxScaler(), preprocessing.Normalizer(), preprocessing.RobustScaler(), None]

# Train and test classifiers with each scaler
for scaler in scalers:
    if scaler is not None:
        X_scaled = scaler.fit_transform(X)
        print(f"\nScaling with {type(scaler).__name__}:")

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        dtc = DecisionTreeClassifier().fit(X_train, y_train)
        lda = LinearDiscriminantAnalysis().fit(X_train, y_train)
        gnb = GaussianNB().fit(X_train, y_train)
        svm = SVC().fit(X_train, y_train)

        kmeans = KMeans().fit(X_train)  # fit the KMeans model on X_train only, without labels

        print(f"Accuracy of Decision Tree classifier: {dtc.score(X_test, y_test)}")
        print(f"Accuracy of Linear Discriminant Analysis classifier: {lda.score(X_test, y_test)}")
        print(f"Accuracy of Gaussian Naive Bayes classifier: {gnb.score(X_test, y_test)}")
        print(f"Accuracy of Support Vector Machine classifier: {svm.score(X_test, y_test)}")

        # Compute the inertia score on X_test
        inertia = kmeans.transform(X_test).min(axis=1).sum()

        print(f"Inertia score of KMeans clustering: {inertia}\n")

        # Train and test the classifiers
        '''dtc = DecisionTreeClassifier().fit(X_train, y_train)
        lda = LinearDiscriminantAnalysis().fit(X_train, y_train)
        gnb = GaussianNB().fit(X_train, y_train)
        svm = SVC().fit(X_train, y_train)
        kmeans = KMeans().fit(X_train, y_train)


        # Print the accuracy of the classifiers
        print(f"Accuracy of Decision Tree classifier: {dtc.score(X_test, y_test)}")
        print(f"Accuracy of Linear Discriminant Analysis classifier: {lda.score(X_test, y_test)}")
        print(f"Accuracy of Gaussian Naive Bayes classifier: {gnb.score(X_test, y_test)}")
        print(f"Accuracy of Support Vector Machine classifier: {svm.score(X_test, y_test)}")
        print(f"Accuracy of KMeans classifier: {kmeans.score(X_test, y_test)}\n")'''
        
    else:
        print("\nNo scaling:")

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        dtc = DecisionTreeClassifier().fit(X_train, y_train)
        lda = LinearDiscriminantAnalysis().fit(X_train, y_train)
        gnb = GaussianNB().fit(X_train, y_train)
        svm = SVC().fit(X_train, y_train)

        kmeans = KMeans().fit(X_train)  # fit the KMeans model on X_train only, without labels

        print(f"Accuracy of Decision Tree classifier: {dtc.score(X_test, y_test)}")
        print(f"Accuracy of Linear Discriminant Analysis classifier: {lda.score(X_test, y_test)}")
        print(f"Accuracy of Gaussian Naive Bayes classifier: {gnb.score(X_test, y_test)}")
        print(f"Accuracy of Support Vector Machine classifier: {svm.score(X_test, y_test)}")

        # Compute the inertia score on X_test
        inertia = kmeans.transform(X_test).min(axis=1).sum()

        print(f"Inertia score of KMeans clustering: {inertia}\n")
