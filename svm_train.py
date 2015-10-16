from scikits.audiolab import wavread, play
import pickle
from scikits.talkbox.features import mfcc
import numpy as np
from sklearn import svm
from sklearn.externals import joblib
from sklearn.linear_model.logistic import LogisticRegression
from time import time
import logging
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(cm,target_names, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(target_names))
    plt.xticks(tick_marks, target_names, rotation=45)
    plt.yticks(tick_marks, target_names)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


#Load positive features
f = open( "models/train_data_pos_mfcc.pkl", "rb" )
X = pickle.load(f)
print "Checking"
for i in range(0,len(X)):
    if (np.isnan(X[i]).any()):
        print i
Y = np.ones(len(X))

#Load negative features
f = open( "models/train_data_neg_mfcc.pkl", "rb" )
X_ = pickle.load(f)
X = X + X_
print "Checking"
for i in range(0,len(X)):
    if (np.isnan(X[i]).any()):
        print i
Y = np.concatenate([Y,np.zeros(len(X_))]) 

#Split features to train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25)

print "Starting Training"
clf = svm.SVC(gamma=0.001, C=1500.)
clf.fit(X, Y)  
print "Saving to disk"
joblib.dump(clf,"models/final_classifier_svm.pkl")
print("Predicting sound segments with SVC")
t0 = time()
y_pred = clf.predict(X_test)
print("done in %0.3fs" % (time() - t0))
print(classification_report(y_test, y_pred,target_names=['Non-Laughter','Laughter']))
cm = confusion_matrix(y_test, y_pred)

# Normalize the confusion matrix by row (i.e by the number of samples
# in each class)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print('Normalized confusion matrix')
print(cm_normalized)
plt.figure()
plot_confusion_matrix(cm_normalized, title='Normalized confusion matrix',target_names=['Non-Laughter','Laughter'])
plt.show()
