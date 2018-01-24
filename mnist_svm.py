from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split
from sklearn import svm, metrics

#load mnist dataset
mnist = fetch_mldata('MNIST original')


X, y = mnist["data"], mnist["target"]

#Split the dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42)

#SVM classifier
classifier = svm.LinearSVC()

classifier.fit(X_train, y_train)

predicted = classifier.predict(X_test)
#print(predicted)

print("Classification report for classifier %s:n%sn" % (classifier, metrics.classification_report(y_test, predicted)))

print("Confusion matrix:n%s" % metrics.confusion_matrix(y_test, predicted))