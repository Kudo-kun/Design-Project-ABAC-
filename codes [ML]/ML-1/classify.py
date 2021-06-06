from numpy import array
import matplotlib.pyplot as plt
from ml_models import Classifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
plt.style.use("ggplot")

def data_preprocessor(fname):
    uvs, ovs, Np = 16, 20, 1
    with open(fname, 'r') as f:
        rules = f.read().split('\n')

    X,  Y = [], []
    for rule in rules:
        (UA, OA, P) = rule.split(';')
        temp_u, temp_o = [0]*uvs, [0]*ovs
        for i in UA.split(','):
            temp_u[int(i)] = 1
        for i in OA.split(','):
            temp_o[int(i)] = 1
        X.append(temp_u + temp_o)
        # X.append([int(a) for a in UA.split(',')] + [int(a) for a in OA.split(',')])
        Y.append(int(P))
    return (array(X), array(Y))


X, Y = data_preprocessor("final_data.txt")
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.2)

models = ["SVM", "LR", "NB", "DT", "RF", "xgboost", "gradboost", "adaboost", "MLP"]
for model in models:
    print(f"[INFO] Training model: {model}")
    clf = Classifier(model)
    clf.fit(Xtrain, Ytrain)
    pred = clf.predict(Xtest)
    acc = accuracy_score(Ytest, pred)
    pre = precision_score(Ytest, pred)
    rec = recall_score(Ytest, pred)
    print(f"accuracy: {acc:.2f}\nprecision: {pre:.2f}\nrecall: {rec:.2f}\n")