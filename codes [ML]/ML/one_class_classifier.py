import argparse
import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


parser = argparse.ArgumentParser()
parser.add_argument("-i", type=str, help="input file for classification")
args = parser.parse_args()

models_dict = {
    "OneClassCSVM": OneClassSVM(),
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


X, Y = data_preprocessor(args.i)
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.2)
# minority class is the outlier
# train with majority class
Xtrain = Xtrain[Ytrain == -1]

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