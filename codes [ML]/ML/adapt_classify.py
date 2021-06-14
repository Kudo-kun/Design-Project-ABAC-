import numpy as np
from ml_models import models_dict
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


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


Xtrain, Ytrain = data_preprocessor("abac-v12.txt")
Xtest, Ytest = data_preprocessor("abac-v3.txt")
Xtrain, Ytrain = RandomOverSampler().fit_resample(Xtrain, Ytrain)

for (name, clf) in models_dict.items():
    print(f"[INFO] Training model: {name}")
    clf.fit(Xtrain, Ytrain)
    pred = clf.predict(Xtest)
    acc = accuracy_score(Ytest, pred)
    pre = precision_score(Ytest, pred)
    rec = recall_score(Ytest, pred)
    f1 = f1_score(Ytest, pred)
    print(f"accuracy: {acc:.2f}\nprecision: {pre:.2f}\nrecall: {rec:.2f}\nfscore: {f1:.2f}\n")