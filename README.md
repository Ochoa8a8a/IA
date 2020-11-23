# IA
To implement the decision tree classifier I used an online tutorial, although I used entropy instead of gini for the data loss calculations. The function is simple, it Iterates all the available columns searching for the one with the best information gain, the way this works is as follows:
Function getEntropy():
If the column has more than 3 different values, it’ll iterate each one, calculating the entropy, to find the best value to split the column into two, all are numeric values.
If the column has 3 or less different values it’ll calculate the entropy of each different value.
Once all columns have been checked the system knows which is the best column information gain and the best split for that column values.
Function buildTree():
This function calls the prior function getEntropy() for each branch, and once it chooses the best info gain generates a sub branch with a subset of the dataset and calls getEntropy() again recursively until there’s no more info to gain.
To compare the results of my Decision Tree Classifier implementation I used sickit learn library to implement a decision tree with the same Titanic datset from kaggle and got the following results:
My decision tree classifier:
56.69 % accuracy
Sci-Kit Learn classifier:
68.42 % accuracy 
That’s an accuracy of 82.86 % compared to the sci-kit classifier.
