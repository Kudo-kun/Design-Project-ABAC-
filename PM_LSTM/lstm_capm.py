import numpy as np
import tensorflow as tf
from bisect import bisect_left
from tensorflow.keras.models import Model 
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.layers import Dense, LSTM, Embedding, Input


def get_vocab(fname):

    f = open(fname, 'r')
    data = f.read().split('\n')
    data.sort()
    f.close()
    return data


user_attr = get_vocab("user_attr.txt")
obj_attr = get_vocab("obj_attr.txt")
f = open("rules.txt")
rules = f.read().split('\n')
f.close()

Xu, Xo, Y = [], [], []
# Permissions are of the RSC (read-set-correct)
# R, S and C take binary values (0 or 1)
# Hence, in decimal, permissions lie between 0-7
for rule in rules:
    (UA, OA, p) = rule.split(';')
    Y.append(int(p))
    Xu.append([bisect_left(user_attr, x)+1 for x in UA.split(' ')])
    Xo.append([bisect_left(obj_attr, x)+1 for x in OA.split(' ')])
Xu = tf.convert_to_tensor(sequence.pad_sequences(Xu))
Xo = tf.convert_to_tensor(sequence.pad_sequences(Xo))
Y = tf.convert_to_tensor(Y)

# generate two encoder models to get the
# essence of the attributes of the user and object
# use those to cell state outputs to determine the
# permission value.
user_vocab_size = len(user_attr)+1
obj_vocab_size = len(obj_attr)+1
inpu = Input(shape=(None, ))
outu = Embedding(user_vocab_size, 16)(inpu)
_, _, cu = LSTM(16, return_state=True)(outu)
inpo = Input(shape=(None, ))
outo = Embedding(obj_vocab_size, 16)(inpo)
_, _, co = LSTM(16, return_state=True)(outo)
out = tf.concat([cu, co], axis=1)
out = Dense(8, activation="softmax")(out)
model = Model(inputs=[inpu, inpo], outputs=out)

# train the model (pass train data)
# works better and faster with GPU
model.compile(optimizer=Adam(learning_rate=0.01),
              loss="sparse_categorical_crossentropy", 
              metrics=["accuracy"])
model.fit([Xu, Xo], Y, epochs=10, batch_size=1)

# test the model (pass test data)
# works better and faster with GPU
model.evaluate([Xu, Xo], Y)
preds = model.predict([Xu, Xo])
class_pred = np.argmax(preds, axis=-1)
print(class_pred)