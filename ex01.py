# NOTE sklearn is a pupolar python library for ML.
from sklearn import tree # NOTE - tree is a module which provides tools to build Decision tree.
from sklearn.model_selection import cross_val_score # NOTE - model_selection is a moudle which provides tools to split data and to evaluate performance of models. NOTE cross_val_score preforms cross validation which is a method for assessing the performance of a model by dividing the data into different groups
import matplotlib.pyplot as plt # NOTE a model which allowing to create all sorts of graphs.
from sklearn import datasets # NOTE - loads the sklearn data sets.


iris = datasets.load_iris() # NOTE - imports the Iris data set.

#mylist = []
#do loop
clf = tree.DecisionTreeClassifier()
clf.max_depth = 5
clf.criterion = 'entropy'
clf = clf.fit(iris.data, iris.target)
print("Decision Tree: ")
accuracy = cross_val_score(clf, iris.data, iris.target, scoring='accuracy', cv=10)
print("Average Accuracy of DT with depth ", clf.max_depth, " is: ",
round(accuracy.mean(),3))

#mylist.append(accuracy.mean()) loop, can be used to plot later…
precision = cross_val_score(clf, iris.data, iris.target, scoring='precision_weighted',
cv=10)
print("Average precision_weighted of DT with depth ", clf.max_depth, " is: ",
round(precision.mean(),3))

#X = range(10)
#plt.plot(X, [x * x for x in X])
#plt.xlabel("This is the X axis")
#plt.ylabel("This is the Y axis")
#plt.show() 