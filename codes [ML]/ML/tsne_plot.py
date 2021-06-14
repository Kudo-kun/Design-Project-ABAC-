import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

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
    return (np.array(X), np.array(Y))


X, Y = data_preprocessor(args.i)

tsne = TSNE(n_components=2)
tsne_obj = tsne.fit_transform(X)
plt.figure(figsize=(16,10))
plt.scatter(x=tsne_obj[:,0], 
            y=tsne_obj[:,1], 
            marker='+', c=Y, 
            cmap="jet")
plt.show()


tsne = TSNE(n_components=3)
tsne_obj = tsne.fit_transform(X)
ax = plt.figure(figsize=(16,10)).add_subplot(projection="3d")
ax.scatter(xs=tsne_obj[:,0], 
           ys=tsne_obj[:,1], 
           zs=tsne_obj[:,2],
           marker='+', c=Y, 
           cmap="jet")
plt.show()