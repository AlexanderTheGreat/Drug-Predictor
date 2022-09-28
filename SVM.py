from sklearn import svm
from sklearn.model_selection import GridSearchCV


class SVM(object):
    def __init__(self, newTrainingVector, ratingsList, newTestingVector):
        self.trainingVector, self.ratingsList, self.testingVector = newTrainingVector, ratingsList, newTestingVector

    def predict(self):
        # clf = svm.SVC()                                     # Creating SVM classifier
        # clf.fit(self.trainingVector, self.ratingsList)      # Training decision tree classifier
        # predictedValues = clf.predict(self.testingVector)   # Placing predictions into list

        svc = svm.SVC(C=100000000.0)  # Creating SVM classifier
        clf = GridSearchCV(svc, {'kernel': ['rbf', 'linear'], 'C': [1, 10], 'gamma': [0.001, 0.01, 0.1, 0.2, 0.5]})
        clfResults = clf.fit(self.trainingVector, self.ratingsList)  # Training decision tree classifier
        bestClf = clfResults.best_estimator_
        predictedValues = bestClf.predict(self.testingVector)  # Placing predictions into list

        print("Best: {0}, using {1}".format(clfResults.cv_results_['mean_test_score'], clfResults.best_params_))

        return predictedValues
