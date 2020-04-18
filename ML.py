import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

fl=pd.read_csv('RandomData.csv')

#print(fl.head(3))
#print(fl.columns[0])
#print(fl.columns[-1])
#del fl['Unnamed: 32']
#print(fl.columns)
#print(fl.columns[:12])

data = fl.iloc[:,:6]
#print(data.info())

ip=data.iloc[:,[1,2,3,4]]
op=data.iloc[:,5]

"""
plt.figure(figsize=(10,10))
sns.heatmap(ip.corr(),cmap='coolwarm')
plt.show()
c_dict =lambda colr: 'red' if colr=='M' else 'blue'
print(c_dict('M'))
cmap=op.map(c_dict)
print(op[:20])
print(cmap[:20])
plt.figure(figsize=(10,10))

sm=pd.plotting.scatter_matrix(ip, c=cmap)
plt.show()

print(ip.columns)
print(ip.corr())

data.set_index('ID', inplace=True)
ip=ip.set_index('ID')
stats= ip.describe().T
mu= stats['mean']
sigma=stats['std']
normal_ip=(ip-mu)/sigma
print(normal_ip.head())
outs=open('outlier.csv','wt')
col_names=ip.columns

digit_dict={'D':1, 'S':0, 3.14:'abs', False:4}
print(digit_dict[False])
new_ops= lambda x: digit_dict.get(x)
op=op.apply(new_ops)
"""
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
#cvs=[]
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

