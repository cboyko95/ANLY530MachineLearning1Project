# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 23:09:02 2021

@author: cboyk
"""
# Packages
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier 
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale 
from sklearn.decomposition import PCA 

# Reading in data
infile = "C:\\Users\\cboyk\\OneDrive\\Desktop\\HU Analytics\\530 Machine Learning I\\Final Project\\Student_Performance_Portuguese_RECODED.csv"
portugese = pd.read_csv(infile) 
portugese.info()

# Dropping G1 and G2
portugese = portugese.drop(["G1", "G2"], axis = 1)

# Recoding G3
failed = portugese.G3 < 10
passed = portugese.G3 >= 10
portugese.loc[passed, 'G3'] = 1
portugese.loc[failed, 'G3'] = 0

# Splitting into test and train sets (Runs with no preprocessing)
Y = portugese.G3
X = portugese.drop(['G3'], axis = 1)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.30, random_state = 52)
Y.value_counts()
Y_test.value_counts()
Y_train.value_counts()

# Classification tree
treemodel = tree.DecisionTreeClassifier() # initializing the model
treemodel = treemodel.fit(X_train, Y_train)
Y_predict = treemodel.predict(X_test)
print(confusion_matrix(Y_test, Y_predict))
print(accuracy_score(Y_test, Y_predict)*100)

# Random Forests
RFmodel = RandomForestClassifier()
RFmodel.fit(X_train, Y_train)
Y_predict = RFmodel.predict(X_test)
print(confusion_matrix(Y_test, Y_predict))
print(accuracy_score(Y_test, Y_predict)*100)
feature_importances=pd.DataFrame(RFmodel.feature_importances_,
                                 index = X_train.columns,
                                  columns=['importance']).sort_values('importance',ascending=False)
feature_importances

# Naive Bayes
Bayesmodel = GaussianNB()
Bayesmodel.fit(X_train, Y_train)
Y_predict = Bayesmodel.predict(X_test)
print(confusion_matrix(Y_test, Y_predict))
print(accuracy_score(Y_test, Y_predict)*100)

# Linear SVM
LinSVMmodel = svm.SVC(kernel = 'linear')
LinSVMmodel.fit(X_train, Y_train)
Y_predict = LinSVMmodel.predict(X_test)
print(confusion_matrix(Y_test, Y_predict))
print(accuracy_score(Y_test, Y_predict)*100)

# Polynomial 2 SVM
Poly2SVMmodel = svm.SVC(kernel = "poly", degree = 2)
Poly2SVMmodel.fit(X_train, Y_train)
Y_predict = Poly2SVMmodel.predict(X_test)
print(confusion_matrix(Y_test, Y_predict))
print(accuracy_score(Y_test, Y_predict)*100)

# Polynomial 3 SVM
Poly3SVMmodel = svm.SVC(kernel = "poly", degree = 3)
Poly3SVMmodel.fit(X_train, Y_train)
Y_predict = Poly3SVMmodel.predict(X_test)
print(confusion_matrix(Y_test, Y_predict))
print(accuracy_score(Y_test, Y_predict)*100)

# Polynomial 4 SVM
Poly4SVMmodel = svm.SVC(kernel = "poly", degree = 4)
Poly4SVMmodel.fit(X_train, Y_train)
Y_predict = Poly4SVMmodel.predict(X_test)
print(confusion_matrix(Y_test, Y_predict))
print(accuracy_score(Y_test, Y_predict)*100)

# Polynomial 5 SVM
Poly5SVMmodel = svm.SVC(kernel = "poly", degree = 5)
Poly5SVMmodel.fit(X_train, Y_train)
Y_predict = Poly5SVMmodel.predict(X_test)
print(confusion_matrix(Y_test, Y_predict))
print(accuracy_score(Y_test, Y_predict)*100)

# RBF SVM
RBFSVMmodel= svm.SVC(kernel = 'rbf')
RBFSVMmodel.fit(X_train, Y_train)
Y_predict = RBFSVMmodel.predict(X_test)
print(confusion_matrix(Y_test, Y_predict))
print(accuracy_score(Y_test, Y_predict)*100)

# KNN
acc = []
for i in range(1,40):
    neigh = KNeighborsClassifier(n_neighbors = i).fit(X_train,Y_train)
    yhat = neigh.predict(X_test)
    acc.append(metrics.accuracy_score(Y_test, yhat))
    
plt.figure(figsize=(10,6))
plt.plot(range(1,40),acc,color = 'blue',linestyle='dashed', 
         marker='o',markerfacecolor='red', markersize=10)
plt.title('accuracy vs. K Value')
plt.xlabel('K')
plt.ylabel('Accuracy')
print("Maximum accuracy:-",max(acc),"at K =",acc.index(max(acc)))
KNNmodel = KNeighborsClassifier(n_neighbors=8) 
KNNmodel.fit(X_train, Y_train) 
Y_predict = KNNmodel.predict(X_test)
print(confusion_matrix(Y_test, Y_predict))
print(accuracy_score(Y_test, Y_predict)*100)

# ____________________________________________________________________________________________

# Splitting into test and train with Scaled
Y = portugese.G3
X = scale(portugese.drop(['G3'], axis = 1))
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.30, random_state = 52)

# Classification tree
treemodel = tree.DecisionTreeClassifier() # initializing the model
treemodel = treemodel.fit(X_train, Y_train)
Y_predict = treemodel.predict(X_test)
print(confusion_matrix(Y_test, Y_predict))
print(accuracy_score(Y_test, Y_predict)*100)

# Random Forests
RFmodel = RandomForestClassifier()
RFmodel.fit(X_train, Y_train)
Y_predict = RFmodel.predict(X_test)
print(confusion_matrix(Y_test, Y_predict))
print(accuracy_score(Y_test, Y_predict)*100)
feature_importances=pd.DataFrame(RFmodel.feature_importances_,
                                 index = X_train.columns,
                                  columns=['importance']).sort_values('importance',ascending=False)
feature_importances

# Naive Bayes
Bayesmodel = GaussianNB()
Bayesmodel.fit(X_train, Y_train)
Y_predict = Bayesmodel.predict(X_test)
print(confusion_matrix(Y_test, Y_predict))
print(accuracy_score(Y_test, Y_predict)*100)

# Linear SVM
LinSVMmodel = svm.SVC(kernel = 'linear')
LinSVMmodel.fit(X_train, Y_train)
Y_predict = LinSVMmodel.predict(X_test)
print(confusion_matrix(Y_test, Y_predict))
print(accuracy_score(Y_test, Y_predict)*100)

# Polynomial 2 SVM
Poly2SVMmodel = svm.SVC(kernel = "poly", degree = 2)
Poly2SVMmodel.fit(X_train, Y_train)
Y_predict = Poly2SVMmodel.predict(X_test)
print(confusion_matrix(Y_test, Y_predict))
print(accuracy_score(Y_test, Y_predict)*100)

# Polynomial 3 SVM
Poly3SVMmodel = svm.SVC(kernel = "poly", degree = 3)
Poly3SVMmodel.fit(X_train, Y_train)
Y_predict = Poly3SVMmodel.predict(X_test)
print(confusion_matrix(Y_test, Y_predict))
print(accuracy_score(Y_test, Y_predict)*100)

# Polynomial 4 SVM
Poly4SVMmodel = svm.SVC(kernel = "poly", degree = 4)
Poly4SVMmodel.fit(X_train, Y_train)
Y_predict = Poly4SVMmodel.predict(X_test)
print(confusion_matrix(Y_test, Y_predict))
print(accuracy_score(Y_test, Y_predict)*100)

# Polynomial 5 SVM
Poly5SVMmodel = svm.SVC(kernel = "poly", degree = 5)
Poly5SVMmodel.fit(X_train, Y_train)
Y_predict = Poly5SVMmodel.predict(X_test)
print(confusion_matrix(Y_test, Y_predict))
print(accuracy_score(Y_test, Y_predict)*100)

# RBF SVM
RBFSVMmodel= svm.SVC(kernel = 'rbf')
RBFSVMmodel.fit(X_train, Y_train)
Y_predict = RBFSVMmodel.predict(X_test)
print(confusion_matrix(Y_test, Y_predict))
print(accuracy_score(Y_test, Y_predict)*100)

# KNN
acc = []
for i in range(1,40):
    neigh = KNeighborsClassifier(n_neighbors = i).fit(X_train,Y_train)
    yhat = neigh.predict(X_test)
    acc.append(metrics.accuracy_score(Y_test, yhat))
    
plt.figure(figsize=(10,6))
plt.plot(range(1,40),acc,color = 'blue',linestyle='dashed', 
         marker='o',markerfacecolor='red', markersize=10)
plt.title('accuracy vs. K Value')
plt.xlabel('K')
plt.ylabel('Accuracy')
print("Maximum accuracy:-",max(acc),"at K =",acc.index(max(acc)))
KNNmodel = KNeighborsClassifier(n_neighbors=14) 
KNNmodel.fit(X_train, Y_train) 
Y_predict = KNNmodel.predict(X_test)
print(confusion_matrix(Y_test, Y_predict))
print(accuracy_score(Y_test, Y_predict)*100)

# _____________________________________________________________________________________________

# PCA
Y = portugese.G3
X = scale(portugese.drop(['G3'], axis = 1))
pca=PCA(0.95)
PC =pca.fit(X)
PC.explained_variance_ratio_
New_X = pca.fit_transform(X)
X_train, X_test, Y_train, Y_test = train_test_split(New_X, Y, test_size = 0.30, random_state = 52)

# Classification tree
treemodel = tree.DecisionTreeClassifier() # initializing the model
treemodel = treemodel.fit(X_train, Y_train)
Y_predict = treemodel.predict(X_test)
print(confusion_matrix(Y_test, Y_predict))
print(accuracy_score(Y_test, Y_predict)*100)

# Random Forests
RFmodel = RandomForestClassifier()
RFmodel.fit(X_train, Y_train)
Y_predict = RFmodel.predict(X_test)
print(confusion_matrix(Y_test, Y_predict))
print(accuracy_score(Y_test, Y_predict)*100)
feature_importances=pd.DataFrame(RFmodel.feature_importances_,
                                 index = X_train.columns,
                                  columns=['importance']).sort_values('importance',ascending=False)
feature_importances

# Naive Bayes
Bayesmodel = GaussianNB()
Bayesmodel.fit(X_train, Y_train)
Y_predict = Bayesmodel.predict(X_test)
print(confusion_matrix(Y_test, Y_predict))
print(accuracy_score(Y_test, Y_predict)*100)

# Linear SVM
LinSVMmodel = svm.SVC(kernel = 'linear')
LinSVMmodel.fit(X_train, Y_train)
Y_predict = LinSVMmodel.predict(X_test)
print(confusion_matrix(Y_test, Y_predict))
print(accuracy_score(Y_test, Y_predict)*100)

# Polynomial 2 SVM
Poly2SVMmodel = svm.SVC(kernel = "poly", degree = 2)
Poly2SVMmodel.fit(X_train, Y_train)
Y_predict = Poly2SVMmodel.predict(X_test)
print(confusion_matrix(Y_test, Y_predict))
print(accuracy_score(Y_test, Y_predict)*100)

# Polynomial 3 SVM
Poly3SVMmodel = svm.SVC(kernel = "poly", degree = 3)
Poly3SVMmodel.fit(X_train, Y_train)
Y_predict = Poly3SVMmodel.predict(X_test)
print(confusion_matrix(Y_test, Y_predict))
print(accuracy_score(Y_test, Y_predict)*100)

# Polynomial 4 SVM
Poly4SVMmodel = svm.SVC(kernel = "poly", degree = 4)
Poly4SVMmodel.fit(X_train, Y_train)
Y_predict = Poly4SVMmodel.predict(X_test)
print(confusion_matrix(Y_test, Y_predict))
print(accuracy_score(Y_test, Y_predict)*100)

# Polynomial 5 SVM
Poly5SVMmodel = svm.SVC(kernel = "poly", degree = 5)
Poly5SVMmodel.fit(X_train, Y_train)
Y_predict = Poly5SVMmodel.predict(X_test)
print(confusion_matrix(Y_test, Y_predict))
print(accuracy_score(Y_test, Y_predict)*100)

# RBF SVM
RBFSVMmodel= svm.SVC(kernel = 'rbf')
RBFSVMmodel.fit(X_train, Y_train)
Y_predict = RBFSVMmodel.predict(X_test)
print(confusion_matrix(Y_test, Y_predict))
print(accuracy_score(Y_test, Y_predict)*100)

# KNN
acc = []
for i in range(1,40):
    neigh = KNeighborsClassifier(n_neighbors = i).fit(X_train,Y_train)
    yhat = neigh.predict(X_test)
    acc.append(metrics.accuracy_score(Y_test, yhat))
    
plt.figure(figsize=(10,6))
plt.plot(range(1,40),acc,color = 'blue',linestyle='dashed', 
         marker='o',markerfacecolor='red', markersize=10)
plt.title('accuracy vs. K Value')
plt.xlabel('K')
plt.ylabel('Accuracy')
print("Maximum accuracy:-",max(acc),"at K =",acc.index(max(acc)))
KNNmodel = KNeighborsClassifier(n_neighbors=14) 
KNNmodel.fit(X_train, Y_train) 
Y_predict = KNNmodel.predict(X_test)
print(confusion_matrix(Y_test, Y_predict))
print(accuracy_score(Y_test, Y_predict)*100)

# ______________________________________________________________________________________________

# Keeping G1
portugese = pd.read_csv(infile) 
portugese = portugese.drop(["G2"], axis = 1)
failed = portugese.G3 < 10
passed = portugese.G3 >= 10
portugese.loc[passed, 'G3'] = 1
portugese.loc[failed, 'G3'] = 0
Y = portugese.G3
X = scale(portugese.drop(['G3'], axis = 1))
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.30, random_state = 52)

# Classification tree
treemodel = tree.DecisionTreeClassifier() # initializing the model
treemodel = treemodel.fit(X_train, Y_train)
Y_predict = treemodel.predict(X_test)
print(confusion_matrix(Y_test, Y_predict))
print(accuracy_score(Y_test, Y_predict)*100)

# Random Forests
RFmodel = RandomForestClassifier()
RFmodel.fit(X_train, Y_train)
Y_predict = RFmodel.predict(X_test)
print(confusion_matrix(Y_test, Y_predict))
print(accuracy_score(Y_test, Y_predict)*100)
feature_importances=pd.DataFrame(RFmodel.feature_importances_,
                                 index = X_train.columns,
                                  columns=['importance']).sort_values('importance',ascending=False)
feature_importances

# Naive Bayes
Bayesmodel = GaussianNB()
Bayesmodel.fit(X_train, Y_train)
Y_predict = Bayesmodel.predict(X_test)
print(confusion_matrix(Y_test, Y_predict))
print(accuracy_score(Y_test, Y_predict)*100)

# Linear SVM
LinSVMmodel = svm.SVC(kernel = 'linear')
LinSVMmodel.fit(X_train, Y_train)
Y_predict = LinSVMmodel.predict(X_test)
print(confusion_matrix(Y_test, Y_predict))
print(accuracy_score(Y_test, Y_predict)*100)

# Polynomial 2 SVM
Poly2SVMmodel = svm.SVC(kernel = "poly", degree = 2)
Poly2SVMmodel.fit(X_train, Y_train)
Y_predict = Poly2SVMmodel.predict(X_test)
print(confusion_matrix(Y_test, Y_predict))
print(accuracy_score(Y_test, Y_predict)*100)

# Polynomial 3 SVM
Poly3SVMmodel = svm.SVC(kernel = "poly", degree = 3)
Poly3SVMmodel.fit(X_train, Y_train)
Y_predict = Poly3SVMmodel.predict(X_test)
print(confusion_matrix(Y_test, Y_predict))
print(accuracy_score(Y_test, Y_predict)*100)

# Polynomial 4 SVM
Poly4SVMmodel = svm.SVC(kernel = "poly", degree = 4)
Poly4SVMmodel.fit(X_train, Y_train)
Y_predict = Poly4SVMmodel.predict(X_test)
print(confusion_matrix(Y_test, Y_predict))
print(accuracy_score(Y_test, Y_predict)*100)

# Polynomial 5 SVM
Poly5SVMmodel = svm.SVC(kernel = "poly", degree = 5)
Poly5SVMmodel.fit(X_train, Y_train)
Y_predict = Poly5SVMmodel.predict(X_test)
print(confusion_matrix(Y_test, Y_predict))
print(accuracy_score(Y_test, Y_predict)*100)

# RBF SVM
RBFSVMmodel= svm.SVC(kernel = 'rbf')
RBFSVMmodel.fit(X_train, Y_train)
Y_predict = RBFSVMmodel.predict(X_test)
print(confusion_matrix(Y_test, Y_predict))
print(accuracy_score(Y_test, Y_predict)*100)

# KNN
acc = []
for i in range(1,40):
    neigh = KNeighborsClassifier(n_neighbors = i).fit(X_train,Y_train)
    yhat = neigh.predict(X_test)
    acc.append(metrics.accuracy_score(Y_test, yhat))
    
plt.figure(figsize=(10,6))
plt.plot(range(1,40),acc,color = 'blue',linestyle='dashed', 
         marker='o',markerfacecolor='red', markersize=10)
plt.title('accuracy vs. K Value')
plt.xlabel('K')
plt.ylabel('Accuracy')
print("Maximum accuracy:-",max(acc),"at K =",acc.index(max(acc)))
KNNmodel = KNeighborsClassifier(n_neighbors=15) 
KNNmodel.fit(X_train, Y_train) 
Y_predict = KNNmodel.predict(X_test)
print(confusion_matrix(Y_test, Y_predict))
print(accuracy_score(Y_test, Y_predict)*100)

# ______________________________________________________________________________________________

# Keeping G1 & G2
portugese = pd.read_csv(infile) 
failed = portugese.G3 < 10
passed = portugese.G3 >= 10
portugese.loc[passed, 'G3'] = 1
portugese.loc[failed, 'G3'] = 0
Y = portugese.G3
X = scale(portugese.drop(['G3'], axis = 1))
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.30, random_state = 52)

# Classification tree
treemodel = tree.DecisionTreeClassifier() # initializing the model
treemodel = treemodel.fit(X_train, Y_train)
Y_predict = treemodel.predict(X_test)
print(confusion_matrix(Y_test, Y_predict))
print(accuracy_score(Y_test, Y_predict)*100)

# Random Forests
RFmodel = RandomForestClassifier()
RFmodel.fit(X_train, Y_train)
Y_predict = RFmodel.predict(X_test)
print(confusion_matrix(Y_test, Y_predict))
print(accuracy_score(Y_test, Y_predict)*100)
feature_importances=pd.DataFrame(RFmodel.feature_importances_,
                                 index = X_train.columns,
                                  columns=['importance']).sort_values('importance',ascending=False)
feature_importances

# Naive Bayes
Bayesmodel = GaussianNB()
Bayesmodel.fit(X_train, Y_train)
Y_predict = Bayesmodel.predict(X_test)
print(confusion_matrix(Y_test, Y_predict))
print(accuracy_score(Y_test, Y_predict)*100)

# Linear SVM
LinSVMmodel = svm.SVC(kernel = 'linear')
LinSVMmodel.fit(X_train, Y_train)
Y_predict = LinSVMmodel.predict(X_test)
print(confusion_matrix(Y_test, Y_predict))
print(accuracy_score(Y_test, Y_predict)*100)

# Polynomial 2 SVM
Poly2SVMmodel = svm.SVC(kernel = "poly", degree = 2)
Poly2SVMmodel.fit(X_train, Y_train)
Y_predict = Poly2SVMmodel.predict(X_test)
print(confusion_matrix(Y_test, Y_predict))
print(accuracy_score(Y_test, Y_predict)*100)

# Polynomial 3 SVM
Poly3SVMmodel = svm.SVC(kernel = "poly", degree = 3)
Poly3SVMmodel.fit(X_train, Y_train)
Y_predict = Poly3SVMmodel.predict(X_test)
print(confusion_matrix(Y_test, Y_predict))
print(accuracy_score(Y_test, Y_predict)*100)

# Polynomial 4 SVM
Poly4SVMmodel = svm.SVC(kernel = "poly", degree = 4)
Poly4SVMmodel.fit(X_train, Y_train)
Y_predict = Poly4SVMmodel.predict(X_test)
print(confusion_matrix(Y_test, Y_predict))
print(accuracy_score(Y_test, Y_predict)*100)

# Polynomial 5 SVM
Poly5SVMmodel = svm.SVC(kernel = "poly", degree = 5)
Poly5SVMmodel.fit(X_train, Y_train)
Y_predict = Poly5SVMmodel.predict(X_test)
print(confusion_matrix(Y_test, Y_predict))
print(accuracy_score(Y_test, Y_predict)*100)

# RBF SVM
RBFSVMmodel= svm.SVC(kernel = 'rbf')
RBFSVMmodel.fit(X_train, Y_train)
Y_predict = RBFSVMmodel.predict(X_test)
print(confusion_matrix(Y_test, Y_predict))
print(accuracy_score(Y_test, Y_predict)*100)

# KNN
acc = []
for i in range(1,40):
    neigh = KNeighborsClassifier(n_neighbors = i).fit(X_train,Y_train)
    yhat = neigh.predict(X_test)
    acc.append(metrics.accuracy_score(Y_test, yhat))
    
plt.figure(figsize=(10,6))
plt.plot(range(1,40),acc,color = 'blue',linestyle='dashed', 
         marker='o',markerfacecolor='red', markersize=10)
plt.title('accuracy vs. K Value')
plt.xlabel('K')
plt.ylabel('Accuracy')
print("Maximum accuracy:-",max(acc),"at K =",acc.index(max(acc)))
KNNmodel = KNeighborsClassifier(n_neighbors=25) 
KNNmodel.fit(X_train, Y_train) 
Y_predict = KNNmodel.predict(X_test)
print(confusion_matrix(Y_test, Y_predict))
print(accuracy_score(Y_test, Y_predict)*100)


