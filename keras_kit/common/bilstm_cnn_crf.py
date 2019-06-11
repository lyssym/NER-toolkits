# _*_ coding: utf-8 _*_

import pickle
from keras.models import Model
from keras.layers import Embedding, Bidirectional, LSTM, TimeDistributed
from keras.layers import Input, Dropout, Conv1D, Dense, concatenate
from keras_contrib.layers import CRF
from keras_contrib.losses import crf_loss, crf_losses
from keras_contrib.metrics import crf_accuracy, crf_viterbi_accuracy, crf_accuracies

from . import process_data

MAX_LEN = 100
EMBED_DIM = 200
BiRNN_UNITS = 200
HALF_WIN_SIZE = 2
DROPOUT_RATE = 0.1
FILTERS = 50
DENSE_DIM = 50


def create_model(train=True):
    if train:
        (train_x, train_y), (test_x, test_y), (vocab, chunk_tags) = process_data.load_data()
    else:
        with open('model/config.pkl', 'rb') as inp:
            (vocab, chunk_tags) = pickle.load(inp)

    input = Input(shape=(MAX_LEN,), dtype='int32', name='input')
    embedding_1 = Embedding(len(vocab), EMBED_DIM, input_length=MAX_LEN, mask_zero=True)(input)
    bilstm = Bidirectional(LSTM(BiRNN_UNITS // 2, return_sequences=True))(embedding_1)
    bilstm_dropout = Dropout(DROPOUT_RATE)(bilstm)

    embedding_2 = Embedding(len(vocab), EMBED_DIM, input_length=MAX_LEN)(input)
    conv = Conv1D(filters=FILTERS, kernel_size=2*HALF_WIN_SIZE+1, padding='same')(embedding_2)
    conv_d = Dropout(DROPOUT_RATE)(conv)
    dense_conv = TimeDistributed(Dense(DENSE_DIM))(conv_d)

    rnn_cnn_merge = concatenate([bilstm_dropout, dense_conv], axis=2)
    dense = TimeDistributed(Dense(len(chunk_tags)))(rnn_cnn_merge)

    crf = CRF(len(chunk_tags), sparse_target=True)
    crf_output = crf(dense)

    model = Model(input=[input], output=[crf_output])
    model.compile(loss=crf_loss, optimizer='adam', metrics=[crf_accuracy])
    model.summary()

    if train:
        return model, (train_x, train_y), (test_x, test_y)
    else:
        return model, (vocab, chunk_tags)
