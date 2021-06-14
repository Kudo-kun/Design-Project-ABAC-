import argparse
import numpy as np
import pandas as pd
from ml_models import models_dict
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

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
    names = ["Designation", "U-Department", "U-Degree", 
             "U-Year", "Type", "O-Department", 
             "O-Degree", "O-Year"]
    df = pd.DataFrame(X, columns=names)
    df["permission"] = Y
    return df


df = data_preprocessor(args.i)
mask = (df["O-Degree"] != 1)
df_train = df[mask]
df_test = df[~mask]

Ytrain, Ytest = np.array(df_train["permission"]), np.array(df_test["permission"])
Xtrain, Xtest = np.array(df_train.drop("permission", 1)), np.array(df_test.drop("permission", 1))
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