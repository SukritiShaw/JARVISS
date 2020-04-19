import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

file =  pd.read_csv('RandomData.csv')

data = file.iloc[:,:6]

ip=data.iloc[:,[1,2,3,4]]
op=data.iloc[:,5]


from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
import time

xtrain,xtest,ytrain,ytest=train_test_split(ip,op,test_size=0.2,random_state=50)
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC , LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
algoname=['StochGradientDescent', 'Support Vector Classifier','LinearSVC',' Decision Tree',' Random Forrest','KNN','GaussianNaiveBayes']
algos=[SGDClassifier(), SVC(), LinearSVC(), DecisionTreeClassifier(), RandomForestClassifier(), KNeighborsClassifier(),GaussianNB()]
time_taken=[]
acc=[]
for i in range (len(algos)):
    starttime=time.time()
    clf=algos[i]
    clf.fit(xtrain, ytrain)
    predictions=clf.predict(xtest)
    stoptime=time.time()
    timetaken=stoptime-starttime
    print('***Current Algo Name =' +algoname[i]+ "***")
    print('Time Taken =')
    print(timetaken)

    time_taken.append(timetaken)

    c_val_score_current=cross_val_score(clf,ip,op,cv=6)
    #print(c_val_score_current)
    accuracy=accuracy_score(predictions, ytest)
    acc.append(accuracy)
    print('Prediction Accuracy =')
    print(accuracy)

