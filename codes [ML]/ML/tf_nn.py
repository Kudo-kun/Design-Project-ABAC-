import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Conv1D, Dropout, Input
from sklearn.metrics import classification_report
from tensorflow.keras.metrics import Precision, Recall


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
    out = Conv1D(7, (3), activation="relu", padding="same")(inp)
    out = Dropout(0.4)(out)
    out = Dense(1, activation="sigmoid")(out)
    model = Model(inp, out)
    return model


Xtrain, Ytrain = data_preprocessor("abac-cat-v1.txt")
Xtest, Ytest = data_preprocessor("test-v1.txt")

model = create_model((1, 8))
model.compile(optimizer="nadam", loss="binary_crossentropy", metrics=['acc', Precision(), Recall()])
model.fit(Xtrain, Ytrain, batch_size=1, epochs=30, validation_data=(Xtest, Ytest), verbose=0)
y_pred = [0 if y < 0.5 else 1 for y in model.predict(Xtest, batch_size=4)]
print(classification_report(Ytest, y_pred))