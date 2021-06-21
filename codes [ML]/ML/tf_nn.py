import numpy as np
from ml_models import models_dict
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tensorflow.keras.metrics import Precision, Recall
import matplotlib.pyplot as plt
plt.style.use("ggplot")


def data_preprocessor(fname):
    with open(fname, 'r') as f:
        rules = f.read().split('\n')

    X,  Y = [], []
    for rule in rules:
        (UA, OA, P) = rule.split(';')
        UA = list(map(int, UA.split(',')))
        OA = list(map(int, OA.split(',')))
        X.append([UA + OA])
        Y.append(int(P))
    return (np.array(X), np.array(Y))


def create_model(inp_shape):
    inp = Input(shape=inp_shape)
    out = Conv1D(5, (3), activation="relu", padding="same")(inp)
    out = Dropout(0.2)(out)
    out = Dense(1, activation="sigmoid")(out)
    model = Model(inp, out)
    return model


Xtrain, Ytrain = data_preprocessor("abac-cat-v3.txt")
Xtest, Ytest = data_preprocessor("test.txt")

model = create_model((1, 8))
model.compile(optimizer="nadam", loss="binary_crossentropy", metrics=["accuracy", Recall(), Precision()])
hist = model.fit(Xtrain, Ytrain, batch_size=4, epochs=30, validation_data=(Xtest, Ytest))

x = range(1, 31)
plt.plot(x, hist.history["accuracy"], color='r', label="training")
plt.plot(x, hist.history["val_accuracy"], color='b', label="validation (test)")
plt.title("Training metric: Accuracy")
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.legend(loc="best")
plt.show()

plt.plot(x, hist.history["precision"], color='r', label="training")
plt.plot(x, hist.history["val_precision"], color='b', label="validation (test)")
plt.title("Training metric: Precision")
plt.xlabel("epochs")
plt.ylabel("precision")
plt.legend(loc="best")
plt.show()

plt.plot(x, hist.history["recall"], color='r', label="training")
plt.plot(x, hist.history["val_recall"], color='b', label="validation (test)")
plt.title("Training metric: Recall")
plt.xlabel("epochs")
plt.ylabel("recall")
plt.legend(loc="best")
plt.show()


f1score_compute = lambda pre, rec : [(2*x*y)/(x+y) if (x+y != 0) else 0 for (x, y) in zip(pre, rec)]
f1score = f1score_compute(hist.history["precision"], hist.history["recall"])
val_f1score = f1score_compute(hist.history["val_precision"], hist.history["val_recall"])
plt.plot(x, f1score, color='r', label="training")
plt.plot(x, val_f1score, color='b', label="validation (test)")
plt.title("Training metric: F1Score")
plt.xlabel("epochs")
plt.ylabel("f1score")
plt.legend(loc="best")
plt.show()