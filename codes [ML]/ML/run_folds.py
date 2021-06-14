import argparse
import numpy as np
from random import randint
from ml_models import models_dict
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
plt.style.use("ggplot")

parser = argparse.ArgumentParser()
parser.add_argument("-i", type=str, help="input file for classification")
args = parser.parse_args()

def data_preprocessor(fname):
    with open(fname, 'r') as f:
        rules = f.read().split('\n')

    X,  Y = [], []
    for rule in rules:
        (UA, OA, P) = rule.split(';')
        UA = [int(a) for a in UA.split(',')]
        OA = [int(a) for a in OA.split(',')]
        X.append(UA + OA)
        Y.append(int(P))
    return (np.array(X), np.array(Y))


def box_plot(scores, title, folds, labels):
    plt.figure(figsize=(16, 10))
    plt.boxplot(scores, labels=labels)
    plt.title(f"{folds}-fold cross validation boxplot for {title}")
    plt.ylabel(title)
    plt.show()


X, Y = data_preprocessor(args.i)
accuracies, precisions, recalls, fscores = [], [], [], []

for (name, clf) in models_dict.items():
    print(f"[INFO] Training model: {name}")
    kfold_gen = KFold(n_splits=7, shuffle=True).split(X, Y)
    current_model_acc = []
    current_model_pre = []
    current_model_rec = []
    current_model_f1 = []
    avg_acc, avg_pre, avg_rec, avg_f1, fold = 0, 0, 0, 0, 0
    
    for (fold, (train, test)) in enumerate(kfold_gen, 1):
        clf.fit(X[train], Y[train])
        ypred = clf.predict(X[test])
        acc = accuracy_score(Y[test], ypred)
        pre = precision_score(Y[test], ypred)
        rec = recall_score(Y[test], ypred)
        f1 = f1_score(Y[test], ypred)
        print(f"[INFO] Training on fold-{fold} | accuracy: {acc:.2f} | precision: {pre:.2f} | recall: {rec:.2f} | fscore: {f1:.2f}")
        
        current_model_acc.append(acc)
        current_model_pre.append(pre)
        current_model_rec.append(rec)
        current_model_f1.append(f1)
        avg_acc += acc
        avg_rec += rec
        avg_pre += pre
        avg_f1 += f1
    
    avg_f1 /= f1
    avg_acc /= fold
    avg_rec /= fold
    avg_pre /= fold
    print(f"[INFO] Average accuracy, precision, recall, fscore over all {fold}-folds: {avg_acc:.2f}, {avg_pre:.2f}, {avg_rec:.2f}, {avg_f1:.2f}\n")
    
    accuracies.append(current_model_acc)
    precisions.append(current_model_pre)
    recalls.append(current_model_rec)
    fscores.append(current_model_f1)
    

box_plot(scores=accuracies, title="accuracy", folds=fold, labels=models_dict.keys())
box_plot(scores=precisions, title="precisions", folds=fold, labels=models_dict.keys())
box_plot(scores=recalls, title="recall", folds=fold, labels=models_dict.keys())
box_plot(scores=fscores, title="fscore", folds=fold, labels=models_dict.keys())