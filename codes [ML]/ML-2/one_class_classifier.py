from numpy import array
import matplotlib.pyplot as plt
from ml_models import Classifier
from sklearn.manifold import TSNE
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
        Y.append(-1 if not int(P) else 1)
    return (array(X), array(Y))


X, Y = data_preprocessor("final_data.txt")

#----------------------------------------------------------------#
tsne = TSNE(n_components=3)
tsne_obj = tsne.fit_transform(X)
ax = plt.figure(figsize=(16,10)).add_subplot(projection='3d')
ax.scatter(xs=tsne_obj[:,0], 
           ys=tsne_obj[:,1], 
           zs=tsne_obj[:,2], 
           c=Y, 
           cmap="jet")

ax.set_xlabel('pca-one')
ax.set_ylabel('pca-two')
ax.set_zlabel('pca-three')
plt.show()

tsne = TSNE(n_components=2)
tsne_obj = tsne.fit_transform(X)
plt.scatter(x=tsne_obj[:,0], 
            y=tsne_obj[:,1], 
            c=Y, 
            cmap="jet")

plt.xlabel('pca-one')
plt.ylabel('pca-two')
plt.show()

#----------------------------------------------------------------#

Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.2)
Xtrain = Xtrain[Ytrain == 1]

models = ["OCSVM", "IF", "EE"]
for model in models:
    print(f"[INFO] Training model: {model}")
    clf = Classifier(model)
    clf.fit(Xtrain)
    pred = clf.predict(Xtest)
    acc = accuracy_score(Ytest, pred)
    pre = precision_score(Ytest, pred)
    rec = recall_score(Ytest, pred)
    print(f"accuracy: {acc:.2f}\nprecision: {pre:.2f}\nrecall: {rec:.2f}")