from sklearn.linear_model import LogisticRegression


class LogisticRegressionClassifier(object):
    def __init__(self, newTrainingVector, ratingsList, newTestingVector):
        self.trainingVector, self.ratingsList, self.testingVector = newTrainingVector, ratingsList, newTestingVector

    def predict(self):
        clf = LogisticRegression()             # Creating and training classifier
        clf.fit(self.trainingVector, self.ratingsList)      # Training decision tree classifier
        predictedValues = clf.predict(self.testingVector)   # Placing predictions into list

        return predictedValues
