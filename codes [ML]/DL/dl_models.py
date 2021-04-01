from tensorflow import concat
from tensorflow.keras.layers import *
from tensorflow.math import reduce_sum
from keras_self_attention import SeqSelfAttention
from tensorflow.keras.models import Model, Sequential


def create_dl_model(model_type, user_vocab_size, obj_vocab_size, nop, out_dim=8, filters=16, kernel=3, lstm_units=8, attn_units=8):
    model_u = Sequential()
    model_o = Sequential()
    
    if model_type == "cnn":
        model_u.add(Conv1D(filters=filters, kernel_size=(kernel), padding="same", activation="relu", input_shape=(1, user_vocab_size)))
        model_u.add(Conv1D(filters=filters//2, kernel_size=(kernel), padding="same", activation="relu"))
        model_u.add(Flatten())
        model_o.add(Conv1D(filters=filters, kernel_size=(kernel), padding="same", activation="relu", input_shape=(1, obj_vocab_size)))
        model_o.add(Conv1D(filters=filters//2, kernel_size=(kernel), padding="same", activation="relu"))
        model_o.add(Flatten())
    elif model_type == "lstm":
        model_u.add(Embedding(user_vocab_size, out_dim))
        model_u.add(LSTM(lstm_units, return_sequences=True))
        model_o.add(Embedding(obj_vocab_size, out_dim))
        model_o.add(LSTM(lstm_units, return_sequences=True))
    elif model_type == "bilstm":
        model_u.add(Embedding(user_vocab_size, out_dim))
        model_u.add(Bidirectional(LSTM(lstm_units, return_sequences=True)))
        model_o.add(Embedding(obj_vocab_size, out_dim))
        model_o.add(Bidirectional(LSTM(lstm_units, return_sequences=True)))
    elif model_type == "attn_bilstm":
        model_u.add(Embedding(user_vocab_size, out_dim))
        model_u.add(Bidirectional(LSTM(lstm_units, return_sequences=True)))
        model_u.add(SeqSelfAttention(attn_units))
        model_o.add(Embedding(obj_vocab_size, out_dim))
        model_o.add(Bidirectional(LSTM(lstm_units, return_sequences=True)))
        model_o.add(SeqSelfAttention(attn_units))

    output = concat([model_u.output, model_o.output], axis=-1)
    dense_units = int((2/3) * output.shape[-1]) + nop
    output = Dense(dense_units, activation="relu")(output)
    output = Dense(nop, activation="sigmoid")(output)
    return Model([model_u.input, model_o.input], output)