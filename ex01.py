# NOTE sklearn is a pupolar python library for ML.
from sklearn import tree # NOTE - tree is a module which provides tools to build Decision tree.
from sklearn.model_selection import cross_val_score # NOTE - model_selection is a moudle which provides tools to split data and to evaluate performance of models. NOTE cross_val_score preforms cross validation which is a method for assessing the performance of a model by dividing the data into different groups
import matplotlib.pyplot as plt # NOTE a model which allowing to create all sorts of graphs.
from sklearn import datasets # NOTE - loads the sklearn data sets.


iris = datasets.load_iris() # NOTE - imports the Iris data set.

#mylist = []
#do loop
clf = tree.DecisionTreeClassifier() # NOTE creates a model of decisions tree.
clf.max_depth = 5 # NOTE define the maximum depth of the tree to be 5. 
clf.criterion = 'entropy' # NOTE the system which by the model will decide how to split the data. NOTE - entropy means by אי הסדר שבקבוצה, NOTE gini means by טוהר של הקבוצה. 

clf = clf.fit(iris.data, iris.target) # NOTE trains the model. iris data are the features of the iris data set, such as Length, Width etc.. NOTE iris.target are the labels of the Iris data set, like Setosa, Virginica, Versicolor etc..

print("Decision Tree: ")

accuracy = cross_val_score(clf, iris.data, iris.target, scoring='accuracy', cv=10) # NOTE defined what this function does in the import phase. NOTE scoring - setting the measurement of model to be by accuracy (number of correct predication / total number of predications).
                                                                                   # NOTE the functions returns a list of 10 values - one value for each iteration. (the data is being split into 10 parts - why 10? because of the cv=10 parameter.), 9 parts are used for training and 1 for testing.

print("Average Accuracy of DT with depth ", clf.max_depth, " is: ",round(accuracy.mean(),3)) # NOTE printing and calculating the average of all the 10 values from the cross_val_score funcm and rounding the result to 3 digits after the dot.

#mylist.append(accuracy.mean()) loop, can be used to plot later…
precision = cross_val_score(clf, iris.data, iris.target, scoring='precision_weighted', cv=10) # NOTE I defined what this function does in the import phase. NOTE Scoring = 'precision_weighted' setting the measurement of the model to be by precision (how much of the positive predications of the model where actually true)
print("Average precision_weighted of DT with depth ", clf.max_depth, " is: ", round(precision.mean(),3))

# NOTE - creating a simple graph according to the given data
X = range(10) # NOTE - Generate a range of values from 0 to 9
plt.plot(X, [x * x for x in X]) # NOTE draws a graph in which the X Axis describes the values of x. NOTE and the Y axis describes the values of of (x^2).
plt.xlabel("This is the X axis") # NOTE sets the description of the X axis.
plt.ylabel("This is the Y axis") # NOTE sets the description of the Y axis.
plt.show() 