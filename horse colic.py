# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#models--------------------------------svm,random forest, decision tree,knn,adaboost,
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler

data=np.genfromtxt(fname='horse-colic.data',delimiter=' ',dtype=float)
#print(data)

#print(data.head())
#print(data.isna().sum())

#deleting hospital number


data=pd.DataFrame(data)
data.fillna(data.median(),inplace=True)
data.drop(2,axis=1,inplace=True)
data=data[0:].values
data=pd.DataFrame(data)
#graphs
sns.countplot(data[21])
plt.xlabel('survival')
plt.show()
plt.figure(figsize=(20,10))
sns.heatmap(data.corr(),annot=True,fmt='.2f',cmap='Blues')

#seprate data into features and target
data=pd.DataFrame(data)
X=data[:]
X.drop(21,axis=1,inplace=True)
X=X[0:].values
Y=data[21].values

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.30,random_state=10)
std = StandardScaler()
X_train = pd.DataFrame(std.fit_transform(X_train))
X_test = pd.DataFrame(std.transform(X_test))


from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


#logistic regression
model = LogisticRegression()
model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_pred, Y_test)
sns.set(font_scale=1.3)
sns.heatmap(cm, annot=True)
plt.show()


#Decision tree classifier
model = DecisionTreeClassifier()
model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_pred, Y_test)
sns.set(font_scale=1.3)
sns.heatmap(cm, annot=True)
plt.show()

#SVC
model = SVC(gamma=0.001)
model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)
#print('Acurracy of ' + k + ' is {0:.2f}'.format(accuracy_score(Y_pred, Y_test)*100))
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_pred, Y_test)
sns.set(font_scale=1.3)
sns.heatmap(cm, annot=True)
plt.show()

#Kneighbour Classifier
model = KNeighborsClassifier(n_neighbors=2)
model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_pred, Y_test)
sns.set(font_scale=1.3)
sns.heatmap(cm, annot=True)
plt.show()


#random forest classifier
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)
#print('Acurracy of '  + ' is {0:.2f}'.format(accuracy_score(Y_pred, Y_test)*100))
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_pred, Y_test)
sns.set(font_scale=1.3)
sns.heatmap(cm, annot=True)
plt.show()

# accuracy of all models together
algo = [LogisticRegression(),DecisionTreeClassifier(),RandomForestClassifier(n_estimators=100),
        SVC(gamma=0.001),
        KNeighborsClassifier(n_neighbors=10)]

for k in algo:
    model = k
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)
    print('Acurracy of ' + str(model).split('(')[0] + ' is {0:.2f}'.format(accuracy_score(Y_pred, Y_test)*100))
    
    
#print("Prediction value " + str(model.predict([X_test[3]])))
#print("Real value " + str(Y_test[3]))
    



#hyper tuning decision tree
clf=DecisionTreeClassifier(random_state=0)
parameters={'max_depth':[1,3,5,7,9,11],
            'class_weight':[{1:0.7,2:0.2,3:0.1},{1:5,2:3,3:1}],
           'max_leaf_nodes':[2,3,5,7],
           'min_samples_leaf':[2,3,5,7,11,15],
           'max_features':['auto','sqrt','log2'],
           'min_samples_split':[2,3,6,11]}
grid_obj=GridSearchCV(clf,parameters,scoring='accuracy',verbose=1)
grid_fit=grid_obj.fit(X_train,Y_train)
best_clf=grid_fit.best_estimator_
print(best_clf)
best_clf.fit(X_train, Y_train)
Y_pred = best_clf.predict(X_test)
print('Acurracy of ' + str(best_clf).split('(')[0] + ' is {0:.2f}'.format(accuracy_score(Y_pred, Y_test)*100))
    

#hyprt tuning randomforest
clf=RandomForestClassifier(random_state=0,n_estimators=100)
parameters={'max_depth':[1,3,5,7,9,11],
            'class_weight':[{1:0.7,2:0.2,3:0.1},{1:5,2:3,3:1}],
           'max_leaf_nodes':[2,3,5,7],
           'min_samples_leaf':[2,3,5,7,11,15],
           #'max_features':['auto','sqrt','log2'],
           'min_samples_split':[6,11,13,15]}
grid_obj=GridSearchCV(clf,parameters,scoring='accuracy',verbose=1)
grid_fit=grid_obj.fit(X_train,Y_train)
best_clf=grid_fit.best_estimator_
print(best_clf)
best_clf.fit(X_train, Y_train)
Y_pred = best_clf.predict(X_test)
print('Acurracy of ' + str(best_clf).split('(')[0] + ' is {0:.2f}'.format(accuracy_score(Y_pred, Y_test)*100))
    
    
