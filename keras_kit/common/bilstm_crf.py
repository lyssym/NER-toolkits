# _*_ coding: utf-8 _*_

import pickle
from keras.models import Sequential
from keras.layers import Embedding, Bidirectional, LSTM
from keras_contrib.layers import CRF
from keras_contrib.losses import crf_loss, crf_losses
from keras_contrib.metrics import crf_accuracy, crf_viterbi_accuracy, crf_accuracies

from . import process_data


EMBED_DIM = 200
BiRNN_UNITS = 200


def create_model(train=True):
    if train:
        (train_x, train_y), (test_x, test_y), (vocab, chunk_tags) = process_data.load_data()
    else:
        with open('model/config.pkl', 'rb') as inp:
            (vocab, chunk_tags) = pickle.load(inp)

    model = Sequential()
    model.add(Embedding(len(vocab), EMBED_DIM, mask_zero=True))  # Random embedding
    model.add(Bidirectional(LSTM(BiRNN_UNITS // 2, return_sequences=True)))
    crf = CRF(len(chunk_tags), sparse_target=True)
    model.add(crf)
    model.summary()
    model.compile('adam', loss=crf_loss, metrics=[crf_accuracy])

    if train:
        return model, (train_x, train_y), (test_x, test_y)
    else:
        return model, (vocab, chunk_tags)
