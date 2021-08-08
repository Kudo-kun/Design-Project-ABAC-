import numpy as np
from csv import writer
from ml_models import models_dict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def data_preprocessor(fname, modify=None):
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


def record_misclassifications(Xtest, Ytest, pred, fname):
    fields = ["Designation", "Department", "Degree", "Year", "Type", "Department", "Degree", "Year", "True_Perm", "Pred_Perm"]
    LUT = [["X", "stu", "prof", "officer", "vis_prof", "adj_prof"],
           ["X", "cse", "ece", "me", "ce", "accounts", "IT", "eni", "pha", "chem"],
           ["X", "btech", "mtech", "BE", "ME", "NA"],
           ["X", "1", "2", "3", "4", "NA"],
           ["X", "asgn", "q_pr", "std_mat", "attdn", "grade_book", "stu_rec", "off_rec", "quiz", "proj", "dept_bud"]]
    f = lambda row: [LUT[0][row[0]], LUT[1][row[1]], LUT[2][row[2]], LUT[3][row[3]], 
                     LUT[4][row[4]], LUT[1][row[5]], LUT[2][row[6]], LUT[3][row[7]]]

    misclassified_pts = [(f(x) + [t] + [y]) for (x, t, y) in zip(Xtest, Ytest, pred) if (y != t)]
    with open(f"./results/{fname}_misclassications.csv", 'w') as csvfile:
        csv_writer = writer(csvfile)
        csv_writer.writerow(fields)
        csv_writer.writerows(misclassified_pts)



Xtrain, Ytrain = data_preprocessor("abac-cat-corrected-v2.txt")
Xtest, Ytest = data_preprocessor("test-v2.txt")

for (name, clf) in models_dict.items():
    print(f"[INFO] Training model: {name}")
    clf.fit(Xtrain, Ytrain)
    pred = clf.predict(Xtest)
    score(Ytrain, clf.predict(Xtrain), label="training metrics")
    score(Ytest, pred, label="testing metrics")
    print("---------------------------------------------------")
    record_misclassifications(Xtest.tolist(), Ytest, pred, fname=name)