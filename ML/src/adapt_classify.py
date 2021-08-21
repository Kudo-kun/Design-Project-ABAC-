import numpy as np
from sys import argv
from csv import writer
from ml_models import models_dict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def data_preprocessor(fname, modify=None):
    print(f"appyling {'no' if modify==None else modify} modification")
    with open(fname, 'r') as f:
        rules = f.read().split('\n')

    X,  Y = [], []
    for rule in rules:
        (UA, OA, P) = rule.split(';')
        UA = list(map(int, UA.split(',')))
        OA = list(map(int, OA.split(',')))
        if modify is None:
            X.append(UA + OA)
        elif modify == "extend":
            extn = [0, 0, 0]
            extn[0] = int(UA[1] == OA[1])
            extn[1] = int(UA[2] == OA[2])
            extn[2] = int(UA[3] == OA[3])
            X.append(UA + OA + extn)
        elif modify == "compress":
            extn = [0, 0, 0]
            extn[0] = int(UA[1] == OA[1])
            extn[1] = int(UA[2] == OA[2])
            extn[2] = int(UA[3] == OA[3])
            X.append([UA[0], OA[0]] + extn)
        Y.append(int(P))
    return (np.array(X), np.array(Y))


def score(Ytest, pred, label):
    acc = accuracy_score(Ytest, pred)
    pre = precision_score(Ytest, pred)
    rec = recall_score(Ytest, pred)
    f1 = f1_score(Ytest, pred)
    print(f"{label}:\naccuracy: {acc:.2f}\nprecision: {pre:.2f}\nrecall: {rec:.2f}\nfscore: {f1:.2f}\n")


modify = None
if len(argv) > 1:
    modify = argv[1]

Xtrain, Ytrain = data_preprocessor("abac-cat-corrected-v1.txt", modify=modify)
Xtest, Ytest = data_preprocessor("test-v1.txt", modify=modify)

for (name, clf) in models_dict.items():
    print(f"\n[INFO] Training model: {name}")
    clf.fit(Xtrain, Ytrain)
    pred = clf.predict(Xtest)
    score(Ytrain, clf.predict(Xtrain), label="training metrics")
    score(Ytest, pred, label="testing metrics")
    print("---------------------------------------------------")