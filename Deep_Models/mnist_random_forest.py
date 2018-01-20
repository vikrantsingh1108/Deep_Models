import sklearn
from sklearn.datasets import fetch_mldata
import numpy
from sklearn import metrics
import random
from numpy import arange
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle
import time

def run():
    
    mnist = fetch_mldata('MNIST original')
    n_train = 60000
    n_test = 10000
    indices = arange(len(mnist.data))
    random.seed(0)
    #print (indices.shape())
    train_id = arange(0, n_train)
    test_id = arange(n_train+1,n_train+n_test)
    
    X_train, y_train = mnist.data[train_id], mnist.target[train_id]
    X_test, y_test = mnist.data[test_id], mnist.target[test_id]
    
    print('Applying learning algorithm...')
    clf = RandomForestClassifier(n_estimators=10,n_jobs=2)
    clf.fit(X_train, y_train)
    # Make a prediction
    print "Making predictions..."
    y_pred = clf.predict(X_test)
    
    # Evaluate the prediction
    print "Evaluating results..."
    print "Precision: \t", metrics.precision_score(y_test, y_pred, average = None)
    print "Recall: \t", metrics.recall_score(y_test, y_pred, average = None)
    print "F1 score: \t", metrics.f1_score(y_test, y_pred, average = None)
    print "Mean accuracy: \t", clf.score(X_test, y_test)

if __name__ == "__main__":
    start_time = time.time()
    results = run()
    end_time = time.time()
    print "Overall running time:", end_time - start_time

