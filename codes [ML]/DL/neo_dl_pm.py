from numpy import array
from matplotlib import pyplot as plt
from dl_models import create_dl_model
from sklearn.model_selection import KFold
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
plt.style.use("ggplot")

EPOCHS = 100
BATCH_SIZE = 32
uvs, ovs, nop = 16, 20, 1
dataset_size = 4380


def data_preprocessor(fname, convert_to_boolean=False, uvs=None, ovs=None):
    f = open(fname, 'r')
    rules = f.read().split('\n')
    f.close()

    Xu, Xo, Y = [], [], []
    for rule in rules:
        (UA, OA, P) = rule.split(';')
        if not convert_to_boolean:
            Xu.append([int(x) for x in UA.split(',')])
            Xo.append([int(x) for x in OA.split(',')])
        else:
            tmp_u, tmp_o = [0]*uvs, [0]*ovs
            for i in UA.split(','):
                tmp_u[int(i)] = 1
            for i in OA.split(','):
                tmp_o[int(i)] = 1
            Xu.append(tmp_u)
            Xo.append(tmp_o)
        Y.append(int(P))
    
    Xu, Xo, Y = array(Xu), array(Xo), array(Y)
    if convert_to_boolean:
        Xu = Xu.reshape((Xu.shape[0], 1, Xu.shape[1]))
        Xo = Xo.reshape((Xo.shape[0], 1, Xo.shape[1]))
    return (Xu, Xo, Y)


all_scores = []
models = ["cnn", "lstm", "bilstm", "attn_bilstm"]
kfold = KFold(n_splits=7, shuffle=True)
callback  = EarlyStopping(monitor="val_binary_accuracy", patience=3)
optimizer = Adam(learning_rate=0.005)

for model_type in models:
    scores, avg_acc, fold = [], 0, 0
    print(f"working on model {model_type}")
    (args1, args2, args3) = (True, uvs, ovs) if (model_type == "cnn") else (False, None, None)
        
    Xu, Xo, Y = data_preprocessor("final_data.txt", 
                                  convert_to_boolean=args1, 
                                  uvs=args2, ovs=args3)
    
    model = create_dl_model(model_type=model_type, 
                            user_vocab_size=uvs, 
                            obj_vocab_size=ovs, nop=nop)
    
    model.compile(optimizer=optimizer, 
                  loss="binary_crossentropy", 
                  metrics=["binary_accuracy"])
    
    for train, test in kfold.split(Xu, Xo, Y):
        Xtrain = [Xu[train], Xo[train]]
        Xtest = [Xu[test], Xo[test]]
        model.fit(Xtrain, Y[train], 
                  epochs=EPOCHS, 
                  batch_size=BATCH_SIZE, 
                  validation_data=(Xtest, Y[test]), 
                  callbacks=[callback], verbose=0)
        
        _, score = model.evaluate(Xtest, Y[test], 
                                  batch_size=BATCH_SIZE, 
                                  verbose=0)
        
        print("[INFO] Training on fold-%d | accuracy: %.3f" % (fold, score))
        scores.append(score)
        avg_acc += score
        fold += 1

    avg_acc /= fold
    all_scores.append(scores)
    print("[INFO] Average accuracy over all %d-folds: %.3f\n\n" % (fold, avg_acc))

plt.boxplot(all_scores, labels=models)
plt.title(f"{fold}-fold cross validation boxplot")
plt.ylabel("accuracy")
plt.show()