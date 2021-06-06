from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier

class Classifier:
    def __init__(self, ctype="svm"):
        if ctype == "SVM":
            self._classifier = SVC(kernel="rbf", probability=True)
        elif ctype == "LR":
            self._classifier = LogisticRegression(n_jobs=-1)
        elif ctype == "NB":
            self._classifier = BernoulliNB()
        elif ctype == "DT":
            self._classifier = DecisionTreeClassifier()
        elif ctype == "RF":
            self._classifier = RandomForestClassifier(n_jobs=-1)
        elif ctype == "xgboost":
            self._classifier = XGBClassifier(use_label_encoder=False, eval_metric="error", n_jobs=-1)
        elif ctype == "gradboost":
            self._classifier = GradientBoostingClassifier(n_estimators=100, learning_rate=0.5)
        elif ctype == "adaboost":
            self._classifier = AdaBoostClassifier(n_estimators=100, learning_rate=0.5)
        elif ctype == "MLP":
            self._classifier = MLPClassifier()

    def fit(self, Xtrain, Ytrain):
        self._classifier.fit(Xtrain, Ytrain)
    
    def predict(self, Xtest):
        return self._classifier.predict(Xtest)

    def predict_proba(self, Xtest):
        return self._classifier.predict_proba(Xtest)
