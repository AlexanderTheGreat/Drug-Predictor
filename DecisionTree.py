from sklearn import tree
from sklearn.model_selection import GridSearchCV

class DecisionTree(object):
    def __init__(self, newTrainingVector, ratingsList, newTestingVector):
        self.trainingVector, self.ratingsList, self.testingVector = newTrainingVector, ratingsList, newTestingVector

    def predict(self):
        # clf = tree.DecisionTreeClassifier()                     # Creating decision tree classifier
        # clf = clf.fit(self.trainingVector, self.ratingsList)    # Training decision tree classifier
        # predictedValues = clf.predict(self.testingVector)       # Placing predictions into list

        dTree = tree.DecisionTreeClassifier()  # Creating SVM classifier
        # print(dTree.get_params().keys())
        param_grid = {"criterion": ["gini", "entropy"],
                      "max_depth": range(1, 10),
                      #"min_samples_leaf": range(1, 10),
                      #"min_samples_split": range(2, 10)
        }

        clf = GridSearchCV(dTree, param_grid=param_grid, cv=10)
        clfResults = clf.fit(self.trainingVector, self.ratingsList)  # Training decision tree classifier
        bestClf = clfResults.best_estimator_
        predictedValues = bestClf.predict(self.testingVector)  # Placing predictions into list
        print("Best: {0}, using {1}".format(clfResults.cv_results_['mean_test_score'], clfResults.best_params_))
        return predictedValues
