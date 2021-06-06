from numpy import array
from ml_models import Classifier
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score
plt.style.use("ggplot")


def data_preprocessor(fname):
    uvs, ovs, Np = 16, 20, 1
    with open(fname, 'r') as f:
        rules = f.read().split('\n')
        f.close()

    X,  Y = [], []
    for rule in rules:
        (UA, OA, P) = rule.split(';')
        temp_u, temp_o = [0]*uvs, [0]*ovs
        for i in UA.split(','):
            temp_u[int(i)] = 1
        for i in OA.split(','):
            temp_o[int(i)] = 1
        X.append(temp_u + temp_o)
        Y.append(int(P))
    return (array(X), array(Y))


def box_plot(scores, title, folds, labels):
    plt.boxplot(scores, labels=labels)
    plt.title(f"{folds}-fold cross validation boxplot for {title}")
    plt.ylabel(title)
    plt.show()


models = ["SVM", "LR", "NB", "DT", "RF", "xgboost", "gradboost", "adaboost", "MLP"]
X, Y = data_preprocessor("final_data.txt")
dataset_size = X.shape[0]
accuracies, precisions, recalls = [], [], []


for model in models:
    print(f"[INFO] Training model: {model}")
    kfold_gen = KFold(n_splits=7, shuffle=True).split(X, Y)
    classifier = Classifier(ctype=model)
    current_model_acc = []
    current_model_pre = []
    current_model_rec = []
    avg_acc, avg_pre, avg_rec, fold = 0, 0, 0, 0
    
    for (fold, (train, test)) in enumerate(kfold_gen, 1):
        classifier.fit(X[train], Y[train])
        ypred = classifier.predict(X[test])
        acc = accuracy_score(Y[test], ypred)
        pre = precision_score(Y[test], ypred)
        rec = recall_score(Y[test], ypred)
        print("[INFO] Training on fold-%d | accuracy: %.3f | precision: %.3f | recall %.3f |" % (fold, acc, pre, rec))
        
        current_model_acc.append(acc)
        current_model_pre.append(pre)
        current_model_rec.append(rec)
        avg_acc += acc
        avg_rec += rec
        avg_pre += pre
    
    avg_acc /= fold
    avg_rec /= fold
    avg_pre /= fold
    print("[INFO] Average accuracy, precision, recall over all %d-fold: %.3f, %.3f, %.3f\n" % (fold, avg_acc, avg_pre, avg_rec))
    accuracies.append(current_model_acc)
    precisions.append(current_model_pre)
    recalls.append(current_model_rec)

    
box_plot(scores=accuracies, title="accuracy", folds=fold, labels=models)
box_plot(scores=precisions, title="precisions", folds=fold, labels=models)
box_plot(scores=recalls, title="recall", folds=fold, labels=models)