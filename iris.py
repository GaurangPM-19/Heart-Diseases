#Training a logistic regression classifier tocheck it ids iris-verginica or not
from sklearn import datasets
import pandas as pd
import numpy as np
iris = pd.read_csv('FLOWER-WEBAPP\iris.csv')
#print(iris.head())
#print(list(iris.keys()))
#['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename']

#get a,b,c,d in  X

# groupby columns a,b,c,d in X
#X = iris.groupby(['a','b','c','d'])
X = iris.iloc[:,0:4]
#print(X)
#mujhe sirf 3rd wala column dedo from iris data mese
#y = (iris["target"] == 2).astype(np.int)
y = iris.iloc[:,4:]
#print(y)
#astype true toh 1 false toh 0
 #we are making a binary classifier to check is 
#the flower is irid-Verginica or not
#print(y)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)


#print(y) gives 0 1 2
print("heficuhqicyqeui")

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#taring a Logistic regression classifier
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression()
clf.fit(X_train, y_train)
import pickle
pickle.dump(clf, open('model.pkl','wb'))
#example = clf.predict([[4.9,3.1,1.5,0.1]])
#print(example)#this will give 0 if 1.6 is not verginica

#using matplotlib to plot
#X_new = np.linspace(0,3,100).reshape(-1,1)#-1,1 will give manu rows 1,-1 will give colums with rows
#linspace will eg(0,3,1000) this will give 10000 point in bet 0 and 3
#print(X_new)


#y_prob = clf.predict_proba(X_new)#if predict() use karenge will give 0 or 1
#print(y_prob)
#import matplotlib.pyplot as plt
#plt.plot(X_new,y_prob[:,1],"g.",label = "verginica")
#plt.show()

