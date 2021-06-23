import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

models_dict = {
    "OneClassCSVM": OneClassSVM(kernel="linear"),
    "IsolationForest": IsolationForest(n_jobs=-1)
}

def data_preprocessor(fname):
    with open(fname, 'r') as f:
        rules = f.read().split('\n')

    X,  Y = [], []
    for rule in rules:
        (UA, OA, P) = rule.split(';')
        UA = list(map(int, UA.split(',')))
        OA = list(map(int, OA.split(',')))
        X.append(UA + OA)
        Y.append(-1 if not int(P) else 1)
    return (np.array(X), np.array(Y))


Xtrain, Ytrain = data_preprocessor("abac-cat-v1.txt")
Xtest, Ytest = data_preprocessor("test-v1.txt")
Xtrain = Xtrain[Ytrain == 1]

for (name, clf) in models_dict.items():
    print(f"[INFO] Training model: {name}")
    clf.fit(Xtrain)
    pred = clf.predict(Xtest)
    acc = accuracy_score(Ytest, pred)
    pre = precision_score(Ytest, pred)
    rec = recall_score(Ytest, pred)
    f1 = f1_score(Ytest, pred)
    conf_mat = confusion_matrix(Ytest, pred)
    print(conf_mat)
    print(f"accuracy: {acc:.2f}\nprecision: {pre:.2f}\nrecall: {rec:.2f}\nfscore: {f1:.2f}\n")