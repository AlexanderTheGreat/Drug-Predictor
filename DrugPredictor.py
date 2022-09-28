from sklearn.feature_extraction.text import TfidfVectorizer
import time
import datetime
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
import DecisionTree
import SVM
import LogisticRegressionClassifier

# pre: receives one specific data line
# post: returns tokenized values
def formatter(data):
    values = [x for x in data.split()]  # split up the values

    return values


print("Initializing...", end="\r")
start = time.time()  # Timing how long the program runs
f_trainData = []  # formatted and cut down list of all the training reviews
f_testData = []  # formatted and cut down list of all the testing reviews
ratingsList = []  # List to hold separated 0 and 1 ratings from train.txt
print("Initializing [Complete]")


# Reading test and training data
print("Reading Data...", end="\r")
trainData = open("src/train.txt", "r")
trainDataList = trainData.readlines()  # itemizes each line in trainData, places the lines into list

testData = open("src/test.txt", "r")
testDataList = testData.readlines()  # itemizes each line in testData, places the lines into list
print("Reading Data [Complete]")


# Tokenizing words
print("Tokenizing Words...", end="\r")
for i in range(len(trainDataList)):
    print(f"Tokenizing Training Data: {i + 1}/{len(trainDataList)}", end="\r")
    ratingsList.append(trainDataList[i][0])  # placing the 0 or 1 at the beginning of the review into ratingsList
    f_trainData.append(" ".join(map(str, formatter(trainDataList[i]))))  # adding formatted values to list

for i in range(len(testDataList)):
    print(f"Tokenizing Testing Data: {i + 1}/{len(testDataList)}", end="\r")
    f_testData.append(" ".join(map(str, formatter(testDataList[i]))))  # adding formatted values to a list
print("Tokenizing [Complete]")


print("Weighing Values...", end="\r")
# use_idf="true" to measure importance of value, max_features = x limits the words we look at down to x amounts
# sublinear tf scaling used to prevent weighing a very common value too highly
tfidf = TfidfVectorizer(use_idf=True, sublinear_tf=True)
trainingVector = tfidf.fit_transform(f_trainData).toarray()
testingVector = tfidf.transform(f_testData).toarray()

print("Weighing Values [Complete]")

print(f"Train Size: {trainingVector.shape}")
print(f"Test Size: {testingVector.shape}")

# performing SVD
print("Performing SVD transformation...", end="\r")
svd = TruncatedSVD(n_components=100)

# transforming data
newTrainingVector = svd.fit_transform(trainingVector)
newTestingVector = svd.transform(testingVector)
print("SVD transformation [Completed]")

print(f"New Train Size: {newTrainingVector.shape}")
print(f"New Test Size: {newTestingVector.shape}")

# charting
# plt.style.use('dark_background')
# x, y = newTrainingVector[:][0], newTrainingVector[:][1]
# x2, y2 = newTestingVector[:][0], newTestingVector[:][1]
# plt.scatter(x2, y2, color="green", marker='o', label='Testing')
# plt.scatter(x, y, color="red", marker='x', label='Training')
# plt.legend(loc='upper left')
# plt.show()

# Classifying Data (Decision Tree, Support Vector Machine, and Logistic Regression)
print("Classifying data...", end="\r")
classificationType = 2

# Decision Tree
if classificationType == 1:
    print("Decision Tree")
    dt = DecisionTree.DecisionTree(newTrainingVector, ratingsList, newTestingVector)
    predictedValues = dt.predict()

# SVM
elif classificationType == 2:
    print("SVM")
    svm = SVM.SVM(newTrainingVector, ratingsList, newTestingVector)
    predictedValues = svm.predict()

# Logistic Regression
else:
    print("Logistic Regression")
    lr = LogisticRegressionClassifier.LogisticRegressionClassifier(newTrainingVector, ratingsList, newTestingVector)
    predictedValues = lr.predict()

print("Classification [Completed]")

with open("format.txt", "w") as format:
    for score in predictedValues:
        format.write(score + "\n")

print("Finished")
print(f"Time to complete: {str(datetime.timedelta(seconds=(time.time() - start)))}")