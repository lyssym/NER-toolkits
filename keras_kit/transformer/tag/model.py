# _*_ coding: utf-8 _*_

import json
import logging
import re
import traceback
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Lock

import keras.backend as K
import numpy as np
from keras import Input, Model, regularizers
from keras.layers import Embedding, Softmax, Dropout, Conv1D, Lambda
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from keras.utils import multi_gpu_model
from keras_contrib.layers import CRF
from keras_contrib.losses import crf_loss
from keras_contrib.metrics import crf_accuracy
from keras_preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer
from keras_transformer.position import TransformerCoordinateEmbedding
from keras_transformer.transformer import TransformerACT, TransformerBlock
from keras_transformer.transformer import gelu

from .utils import load_dictionary


def label_smoothing_loss(y_true, y_pred):
    shape = K.int_shape(y_pred)
    n_class = shape[2]
    eps = 0.1
    y_true = y_true * (1 - eps) + eps / n_class
    return categorical_crossentropy(y_true, y_pred)


def padding_mask(seq_q, seq_k):
    """
    A sentence is filled with 0, which is not what we need to pay attention to
    :param seq_k: shape of [N, T_k], T_k is length of sequence
    :param seq_q: shape of [N, T_q]
    :return: a tensor with shape of [N, T_q, T_k]
    """
    q = K.expand_dims(K.ones_like(seq_q, dtype="float32"), axis=-1)            # [N, T_q, 1]
    k = K.cast(K.expand_dims(K.not_equal(seq_k, 0), axis=1), dtype='float32')  # [N, 1, T_k]
    return K.batch_dot(q, k, axes=[2, 1])


class TFModel:
    def __init__(self,
                 src_vocab_size: int,
                 tgt_vocab_size: int,
                 max_seq_len: int,
                 embedding_size_word: int = 300,
                 model_dim: int = 128,
                 num_filters: int = 128,
                 max_depth: int = 8,
                 num_heads: int = 8,
                 embedding_dropout: float = 0.0,
                 residual_dropout: float = 0.0,
                 attention_dropout: float = 0.0,
                 output_dropout: float = 0.0,
                 confidence_penalty_weight: float = 0.1,
                 l2_reg_penalty: float = 1e-6,
                 compression_window_size: int = None,
                 use_crf: bool = True,
                 optimizer=Adam(),
                 src_tokenizer: Tokenizer = None,
                 tgt_tokenizer: Tokenizer = None,
                 weights_path: str = None,
                 sparse_target: bool = False,
                 num_gpu: int = 1):
        """
        :param src_vocab_size:  源字库大小
        :param tgt_vocab_size:  目标标签数量
        :param max_seq_len:     最大输入长度
        :param model_dim:       Transformer 模型维度
        :param max_depth:       Universal Transformer 深度
        :param num_heads:       多头注意力头数
        :param embedding_dropout: 词嵌入失活率
        :param residual_dropout:  残差失活率
        :param attention_dropout: 注意力失活率
        :param confidence_penalty_weight: confidence_penalty 正则化，仅在禁用CRF时有效
        :param l2_reg_penalty:  l2 正则化率
        :param compression_window_size: 压缩窗口大小
        :param use_crf:     是否使用crf
        :param optimizer:   优化器
        :param src_tokenizer: 源切割器
        :param tgt_tokenizer: 目标切割器
        :param weights_path: 权重路径
        :param num_gpu: 使用gpu数量
        """
        self.optimizer = optimizer
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.max_seq_len = max_seq_len
        self.num_heads = num_heads
        self.max_depth = max_depth
        self.num_gpu = num_gpu
        self.embedding_size_word = embedding_size_word
        self.model_dim = model_dim
        self.num_filters = num_filters
        self.embedding_dropout = embedding_dropout
        self.residual_dropout = residual_dropout
        self.attention_dropout = attention_dropout
        self.output_dropout = output_dropout
        self.confidence_penalty_weight = confidence_penalty_weight
        self.l2_reg_penalty = l2_reg_penalty
        self.compression_window_size = compression_window_size
        self.use_crf = use_crf
        self.sparse_target = sparse_target

        self.model, self.parallel_model = self.__build_model()

        if weights_path is not None:
            try:
                self.model.load_weights(weights_path)
            except Exception as e:
                logging.error(e)
                logging.info("Fail to load weights, create a new model!")

    def __build_model(self):
        assert self.max_depth >= 1, "The parameter max_depth is at least 1"

        src_seq_input = Input(shape=(self.max_seq_len,), dtype="int32", name="src_seq_input")
        mask = Lambda(lambda x: padding_mask(x, x))(src_seq_input)

        emb_output = self.__input(src_seq_input)
        enc_output = self.__encoder(emb_output, mask)

        if self.use_crf:
            crf = CRF(self.tgt_vocab_size + 1,
                      sparse_target=self.sparse_target)
            y_pred = crf(self.__output(enc_output))
        else:
            y_pred = self.__output(enc_output)

        model = Model(inputs=[src_seq_input], outputs=[y_pred])
        parallel_model = model
        if self.num_gpu > 1:
            parallel_model = multi_gpu_model(model, gpus=self.num_gpu)

        if self.use_crf:
            parallel_model.compile(self.optimizer,
                                   loss=crf_loss,
                                   metrics=[crf_accuracy])
        else:
            confidence_penalty = K.mean(
                self.confidence_penalty_weight *
                K.sum(y_pred * K.log(y_pred),
                axis=-1))
            model.add_loss(confidence_penalty)
            parallel_model.compile(optimizer=self.optimizer,
                                   loss=categorical_crossentropy,
                                   metrics=['accuracy'])

        return model, parallel_model

    def __encoder(self, emb_inputs, mask):
        transformer_enc_layer = TransformerBlock(
            name='transformer_enc',
            num_heads=self.num_heads,
            residual_dropout=self.residual_dropout,
            attention_dropout=self.attention_dropout,
            compression_window_size=self.compression_window_size,
            use_masking=False,
            vanilla_wiring=True)
        coordinate_embedding_layer = TransformerCoordinateEmbedding(name="coordinate_emb1",
                                                                    max_transformer_depth=self.max_depth)
        transformer_act_layer = TransformerACT(name='adaptive_computation_time1')

        next_step_input = emb_inputs
        act_output = next_step_input

        for step in range(self.max_depth):
            next_step_input = coordinate_embedding_layer(next_step_input, step=step)
            next_step_input = transformer_enc_layer(next_step_input, padding_mask=mask)
            next_step_input, act_output = transformer_act_layer(next_step_input)

        transformer_act_layer.finalize()
        next_step_input = act_output

        return next_step_input

    def __input(self, src_seq_input):
        embedding_layer = Embedding(self.src_vocab_size + 1, self.embedding_size_word,
                                    input_length=self.max_seq_len,
                                    name='embeddings')

        emb_project_layer = Conv1D(self.model_dim, activation="linear",
                                   kernel_size=1,
                                   name="emb_project")

        emb_dropout_layer = Dropout(self.embedding_dropout, name="emb_dropout")

        emb_output = emb_project_layer(emb_dropout_layer(embedding_layer(src_seq_input)))
        return emb_output

    def __output(self, dec_output):
        output_dropout_layer = Dropout(self.output_dropout)
        output_layer = Conv1D(self.tgt_vocab_size + 1,
                              kernel_size=1,
                              activation=gelu,
                              kernel_regularizer=regularizers.l2(self.l2_reg_penalty),
                              name='output_layer')

        output_softmax_layer = Softmax(name="word_predictions")

        if self.use_crf:
            return output_layer(output_dropout_layer(dec_output))
        else:
            return output_softmax_layer(output_layer(output_dropout_layer(dec_output)))

    def decode_sequences(self, sequences):
        sequences = self._seq_to_matrix(sequences)
        output = self.model.predict_on_batch(sequences)  # [N, -1, chunk_size + 1]
        output = np.argmax(output, axis=2)
        return self.tgt_tokenizer.sequences_to_texts(output)

    def _single_decode(self, args):
        sent, tag = args
        cur_sent, cur_tag = [], []
        tag = tag.split(' ')
        t1, pre_pos = [], None
        for i in range(len(sent)):
            tokens = tag[i].split('-')
            if len(tokens) == 2:
                c, pos = tokens
            else:
                c = 'i'
                pos = "<UNK>"

            word = sent[i]
            if c in 'sb':
                if len(t1) != 0:
                    cur_sent.append(''.join(t1))
                    cur_tag.append(pre_pos)
                t1 = [word]
                pre_pos = pos
            elif c in 'ie':
                t1.append(word)
                pre_pos = pos

        if len(t1) != 0:
            cur_sent.append(''.join(t1))
            cur_tag.append(pre_pos)

        print("sent000: ", cur_sent)
        print("tag000: ", cur_tag)
        return cur_sent, cur_tag

    def decode_texts(self, texts):
        sents = []
        with ThreadPoolExecutor() as executor:
            for text in executor.map(lambda x: list(re.subn("\s+", "", x)[0]), texts):
                sents.append(text)

        sequences = self.src_tokenizer.texts_to_sequences(sents)
        tags = self.decode_sequences(sequences)

        ret = []
        with ThreadPoolExecutor() as executor:
            for cur_sent, cur_tag in executor.map(self._single_decode, zip(sents, tags)):
                ret.append((cur_sent, cur_tag))

        return ret

    def _seq_to_matrix(self, sequences):
        # max_len = len(max(sequences, key=len))
        return pad_sequences(sequences, maxlen=self.max_seq_len, padding="post")

    def get_config(self):
        return {
            'src_vocab_size': self.src_vocab_size,
            'tgt_vocab_size': self.tgt_vocab_size,
            'max_seq_len': self.max_seq_len,
            'max_depth': self.max_depth,
            'model_dim': self.model_dim,
            'embedding_size_word': self.embedding_size_word,
            'confidence_penalty_weight': self.confidence_penalty_weight,
            'l2_reg_penalty': self.l2_reg_penalty,
            'embedding_dropout': self.embedding_dropout,
            'residual_dropout': self.residual_dropout,
            'attention_dropout': self.attention_dropout,
            'compression_window_size': self.compression_window_size,
            'num_heads': self.num_heads,
            'use_crf': self.use_crf
        }

    __singleton = None
    __lock = Lock()

    @staticmethod
    def get_or_create(config, src_dict_path=None,
                      tgt_dict_path=None,
                      weights_path=None,
                      num_gpu=1,
                      optimizer=Adam(),
                      encoding="utf-8"):
        TFModel.__lock.acquire()

        try:
            if TFModel.__singleton is None:
                if type(config) == str:
                    with open(config, encoding=encoding) as file:
                        config = dict(json.load(file))
                elif type(config) == dict:
                    config = config
                else:
                    raise ValueError("Unexpect config type!")

                if src_dict_path is not None:
                    src_tokenizer = load_dictionary(src_dict_path, encoding)
                    config['src_tokenizer'] = src_tokenizer
                if tgt_dict_path is not None:
                    config['tgt_tokenizer'] = load_dictionary(tgt_dict_path, encoding)

                config["num_gpu"] = num_gpu
                config['weights_path'] = weights_path
                config['optimizer'] = optimizer
                TFModel.__singleton = TFModel(**config)
        except Exception:
            traceback.print_exc()
        finally:
            TFModel.__lock.release()
        return TFModel.__singleton


get_or_create = TFModel.get_or_create


def save_config(obj, config_path, encoding="utf-8"):
    with open(config_path, mode="w+", encoding=encoding) as file:
        json.dump(obj.get_config(), file)
