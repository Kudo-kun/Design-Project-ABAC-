import argparse
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
plt.style.use("ggplot")

models_dict = {
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="error", n_jobs=-1),
    "LightGBM": LGBMClassifier(n_jobs=-1),
    "DecisionTree": DecisionTreeClassifier(),
    "GradientBoosting": GradientBoostingClassifier(n_estimators=100, learning_rate=1.0),
    "RandomForest": RandomForestClassifier(n_jobs=-1),
}

parser = argparse.ArgumentParser()
parser.add_argument("-i", type=str, help="input file for classification")
args = parser.parse_args()

def data_preprocessor(fname):
    with open(fname, 'r') as f:
        rules = f.read().split('\n')

    X,  Y = [], []
    for rule in rules:
        (UA, OA, P) = rule.split(';')
        UA = list(map(int, UA.split(',')))
        OA = list(map(int, OA.split(',')))
        X.append(UA + OA)
        Y.append(int(P))
    return (np.array(X), np.array(Y))


X, Y = data_preprocessor(args.i)
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.3)
Xtrain, Ytrain = RandomOverSampler().fit_resample(Xtrain, Ytrain)

for (name, clf) in models_dict.items():
    clf.fit(Xtrain, Ytrain)
    pred = clf.predict(Xtest)
    pred_prob = clf.predict_proba(Xtest)
    fpr, tpr, _ = roc_curve(Ytest, pred_prob[:,1])
    auc = roc_auc_score(Ytest, pred)
    plt.plot(fpr, tpr, label="%s ROC (area = %0.2f)" % (name, auc))
    
plt.plot([0, 1], [0, 1],"r--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("1-Specificity (False Positive Rate)")
plt.ylabel("Sensitivity (True Positive Rate)")
plt.title(f"Receiver Operating Characteristic {args.i}")
plt.legend(loc="best")
stem, _ = args.i.split('.')
plt.savefig(f"ROC plot for {stem}.png")
plt.clf()
