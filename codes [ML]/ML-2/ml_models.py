from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest

class Classifier:

    def __init__(self, ctype="OCSVM"):
        if ctype == "OCSVM":
            self._classifier = OneClassSVM()
        elif ctype == "EE":
            self._classifier = EllipticEnvelope()
        elif ctype == "IF":
            self._classifier = IsolationForest(n_jobs=-1)

    def fit(self, Xtrain):
        self._classifier.fit(Xtrain)
    
    def predict(self, Xtest):
        return self._classifier.predict(Xtest)
