# _*_ coding: utf-8 _*_

import sys
import random

import numpy as np
from itertools import chain
from .config import *
from .utils.data import Data
from .core.cell.lattice_lstm import LatticeLSTMCell

import tensorflow as tf
from tensorflow.python import debug as tf_debug

from sklearn.metrics import confusion_matrix


class LatticeNet(object):
    def __init__(self):
        self.batch_size = BATCH_SIZE
        self.max_char_len = MAX_CHAR_LEN
        self.emb_size = EMB_SIZE
        self.max_lexicon_words_num = MAX_LEXICON_WORDS_NUM
        self.num_units = NUM_UNITS
        self.num_tags = NUM_TAGS
        self.learning_rate = LEARNING_RATE
        self.optimizer = OPTIMIZER
        self.clip = CLIP

        self.gaz_file = GAZ_FILE
        self.char_emb = CHAR_EMB
        self.train_file = TRAIN_FILE
        self.dev_file = DEV_FILE
        self.test_file = TEST_FILE
        self.model_save_path = MODEL_SAVE_PATH

        def my_filter_callable(datum, tensor):
            # A filter that detects zero-valued scalars.
            return len(tensor.shape) == 0 and tensor == 0.0

        self.sess = tf_debug.LocalCLIDebugWrapperSession(tf.Session())
        self.sess.add_tensor_filter('my_filter', my_filter_callable)

        self.sess = tf.Session()
        self.placeholders = {}
        self.epoch = 0
        self.loss = None
        self.global_step = tf.Variable(0, trainable=False)
        self.train_op = None
        self.bichar_emb = None

    def loss_layer(self, project_logits, lengths, labels, name=None):
        """ calculate crf loss
        :param project_logits: [batch_size, num_steps, num_tags]
        :param lengths: [batch_size, num_steps]
        :param labels: [batch_size, num_steps]
        :return: scalar loss
        """
        with tf.variable_scope("crf_loss" if not name else name):
            small = -1000.0
            # pad logits for crf loss
            start_logits = tf.concat(
                [small * tf.ones(shape=[self.batch_size, 1, self.num_tags]),
                 tf.zeros(shape=[self.batch_size, 1, 1])], axis=-1)

            pad_logits = tf.cast(small * tf.ones([self.batch_size, self.max_char_len, 1]), tf.float32)
            logits = tf.concat([project_logits, pad_logits], axis=-1)
            logits = tf.concat([start_logits, logits], axis=1)

            targets = tf.concat(
                [tf.cast(self.num_tags*tf.ones([self.batch_size, 1]), tf.int32), labels], axis=-1)

            self.trans = tf.get_variable(
                "transitions",
                shape=[self.num_tags + 1, self.num_tags + 1],
                initializer=tf.random_uniform_initializer(
                    0.008, 0.15, seed=1311, dtype=tf.float32))

            log_likelihood, self.trans = tf.contrib.crf.crf_log_likelihood(
                inputs=logits,
                tag_indices=targets,
                transition_params=self.trans,
                sequence_lengths=lengths + 1)

            return tf.reduce_sum(-log_likelihood)
            #return tf.reduce_mean(-log_likelihood)

    def create_model(self):
        # print(self.data.pretrain_word_embedding)
        char_embeddings = tf.Variable(self.data.pretrain_word_embedding,
                                      dtype=tf.float32, name="char_embeddings")
        word_embeddings = tf.Variable(self.data.pretrain_gaz_embedding,
                                      dtype=tf.float32, name="word_embeddings")

        char_ids = tf.placeholder(tf.int32, [None, self.max_char_len])
        char_embed = tf.nn.embedding_lookup(char_embeddings, char_ids)

        lexicon_word_ids = tf.placeholder(tf.int32, [None, self.max_char_len,
                                                     self.max_lexicon_words_num])
        word_length_tensor = tf.placeholder(tf.float32, [None, self.max_char_len,
                                                         self.max_lexicon_words_num])

        labels = tf.placeholder(tf.int32, [None, self.max_char_len])

        lexicon_word_ids_reshape = tf.reshape(lexicon_word_ids,
                                              [-1, self.max_char_len * self.max_lexicon_words_num])
        lexicon_word_embed_reshape = tf.nn.embedding_lookup(word_embeddings, lexicon_word_ids_reshape)
        lexicon_word_embed = tf.reshape(lexicon_word_embed_reshape,
                                        [-1, self.max_char_len, self.max_lexicon_words_num, self.emb_size])

        lattice_lstm = LatticeLSTMCell(self.num_units,
                                       self.num_units,
                                       batch_size=self.batch_size,
                                       seq_len=self.max_char_len,
                                       max_lexicon_words_num=self.max_lexicon_words_num,
                                       word_length_tensor=word_length_tensor,
                                       dtype=tf.float32)

        initial_state = lattice_lstm.zero_state(batch_size=self.batch_size, dtype=tf.float32)
        outputs, state = tf.nn.dynamic_rnn(cell=lattice_lstm,
                                           inputs=[char_embed, lexicon_word_embed],
                                           initial_state=initial_state,
                                           dtype=tf.float32)

        # projection:
        W = tf.get_variable("projection_w", [self.num_units, self.num_tags])
        b = tf.get_variable("projection_b", [self.num_tags])
        x_reshape = tf.reshape(outputs, [-1, self.num_units])
        projection = tf.matmul(x_reshape, W) + b

        # -1 to timestep
        self.logits = tf.reshape(projection, [self.batch_size, -1, self.num_tags])
        seq_length = tf.convert_to_tensor(self.batch_size * [self.max_char_len], dtype=tf.int32)

        self.loss = self.loss_layer(self.logits, seq_length, labels)

        with tf.variable_scope("optimizer"):
            optimizer = self.optimizer
            if optimizer == "sgd":
                self.opt = tf.train.GradientDescentOptimizer(self.learning_rate)
            elif optimizer == "adam":
                self.opt = tf.train.AdamOptimizer(self.learning_rate)
            elif optimizer == "adgrad":
                self.opt = tf.train.AdagradOptimizer(self.learning_rate)
            else:
                raise KeyError

            # apply grad clip to avoid gradient explosion
            grads_vars = self.opt.compute_gradients(self.loss)
            capped_grads_vars = map(
                lambda gv: gv if gv[0] is None else [tf.clip_by_value(gv[0], -self.clip, self.clip), gv[1]],
                grads_vars)
            self.train_op = self.opt.apply_gradients(capped_grads_vars, self.global_step)

        # self.train_op = tf.train.AdamOptimizer().minimize(self.loss)

        self.placeholders["char_ids"] = char_ids
        self.placeholders["lexicon_word_ids"] = lexicon_word_ids
        self.placeholders["word_length_tensor"] = word_length_tensor
        self.placeholders["labels"] = labels

    def save_tf_model(self):
        saver = tf.train.Saver()
        saver.save(self.sess, self.model_save_path)

    def restore_tf_model(self):
        saver = tf.train.Saver()
        saver.restore(self.sess, self.model_save_path)

    def train_model(self, data, iters=2000):
        init = tf.global_variables_initializer()
        sess = self.sess
        sess.run(init)

        for iter in range(iters):
            print('iter: ', iter)
            random.shuffle(data.train_Ids)

            train_num = len(data.train_Ids)
            total_batch = train_num // self.batch_size

            for batch_id in range(total_batch):
                start = batch_id * self.batch_size
                end = (batch_id + 1) * self.batch_size

                if end > train_num:
                    end = train_num

                instance = data.train_Ids[start:end]
                if not instance:
                    continue

                self.epoch += 1
                _, char_ids, lexicon_word_ids, word_length_tensor, _, labels = self.batch_with_label(instance)

                # run模型
                feed_dict = {
                    self.placeholders["char_ids"]: char_ids,
                    self.placeholders["lexicon_word_ids"]: lexicon_word_ids,
                    self.placeholders["word_length_tensor"]: word_length_tensor,
                    self.placeholders["labels"]: labels,
                }

                _, losses, step = sess.run([self.train_op, self.loss, self.global_step], feed_dict=feed_dict)

                if self.epoch % 10 == 0:
                    print('*' * 100)
                    print(self.epoch, 'loss', losses)

                    # self.evaluate(sess, data)
                    self.evaluate_line(sess,
                                       ['习', '近', '平', '在', '北', '京', '中', '南', '海', '呼', '吁',
                                        '美', '国', '加', '强', '合', '作', '共', '创', '美', '好', '生', '活'],
                                       data)

    def load_data_and_embedding(self, data):
        data.HP_use_char = False
        data.HP_batch_size = 1
        data.use_bigram = False
        data.gaz_dropout = 0.5
        data.norm_gaz_emb = False
        data.HP_fix_gaz_emb = False

        self.data_initialization(data, self.gaz_file, self.train_file,
                                 self.dev_file, self.test_file)

        data.generate_instance_with_gaz(self.train_file, 'train')
        data.generate_instance_with_gaz(self.dev_file, 'dev')
        data.generate_instance_with_gaz(self.test_file, 'test')

        data.build_word_pretrain_emb(self.char_emb)
        data.build_biword_pretrain_emb(self.bichar_emb)
        data.build_gaz_pretrain_emb(self.gaz_file)

    def data_initialization(self, data, gaz_file, train_file, dev_file, test_file):
        data.build_alphabet(train_file)
        data.build_alphabet(dev_file)
        data.build_alphabet(test_file)

        data.build_gaz_file(gaz_file)

        data.build_gaz_alphabet(train_file)
        data.build_gaz_alphabet(dev_file)
        data.build_gaz_alphabet(test_file)
        data.fix_alphabet()

    def batch_with_label(self, input_batch_list, is_train=True):
        """
        input: list of words, chars and labels, various length.
            [[words,biwords,chars,gaz,labels], [words,biwords,chars,gaz,labels],...]
            words: word ids for one sentence. (batch_size, sent_len)
            chars: char ids for one sentences, various length. (batch_size, sent_len, each_word_length)
        output:
            char_ids: (batch_size, )
            lexicon_word_ids: (batch_size, )
            word_length_tensor: (batch_size, )
            labels: (batch_size, )
        """
        batch_size = len(input_batch_list)
        lengths = [len(sent[0][0:self.max_char_len]) for sent in input_batch_list]
        chars_ids = [sent[0][0:self.max_char_len] for sent in input_batch_list]
        biwords = [sent[1][0:self.max_char_len] for sent in input_batch_list]
        chars_ids_split = [sent[2][0:self.max_char_len] for sent in input_batch_list]
        lexicon_words = [sent[3][0:self.max_char_len] for sent in input_batch_list]

        if is_train:
            target = [sent[4][0:self.max_char_len] for sent in input_batch_list]

        chars_ids = list(map(lambda l: l + [0] * (self.max_char_len - len(l)), chars_ids))
        biwords = list(map(lambda l: l + [0] * (self.max_char_len - len(l)), biwords))

        if is_train:
            labels = list(map(lambda l: l + [0] * (self.max_char_len - len(l)), target))

        lexicon_word_ids = []
        word_length_tensor = []
        for sent in input_batch_list:
            lexicon_word_ids_sent = []
            word_length_tensor_sent = []

            for word_lexicon in sent[3][0:self.max_char_len]:
                word_lexicon_pad = list(map(lambda l:
                                            l + [0] * (self.max_lexicon_words_num - len(l)),
                                            word_lexicon))
                lexicon_word_ids_sent.append(word_lexicon_pad[0][0:self.max_lexicon_words_num])    # id
                word_length_tensor_sent.append(word_lexicon_pad[1][0:self.max_lexicon_words_num])  # length

            lexicon_word_ids.append(lexicon_word_ids_sent)
            word_length_tensor.append(word_length_tensor_sent)

        lexicon_word_ids = list(map(lambda l:
                                    l + [[0] * self.max_lexicon_words_num] * (self.max_char_len - len(l)),
                                    lexicon_word_ids))
        word_length_tensor = list(map(lambda l:
                                      l + [[0] * self.max_lexicon_words_num] * (self.max_char_len - len(l)),
                                      word_length_tensor))

        if is_train:
            return lengths, chars_ids, lexicon_word_ids, word_length_tensor, target, labels

        return lengths, chars_ids, lexicon_word_ids, word_length_tensor

    def decode(self, logits, lengths, transition_matrix):
        """
        :param logits: [batch_size, num_steps, num_tags]float32, logits
        :param lengths: [batch_size]int32, real length of each sequence
        :param transition_matrix: transaction matrix for inference
        :return:
        """
        # inference final labels usa viterbi Algorithm
        paths = []
        small = -1000.0
        start = np.asarray([[small] * self.num_tags + [0]])
        for score, length in zip(logits, lengths):
            score = score[:length]
            pad = small * np.ones([length, 1])

            logits = np.concatenate([score, pad], axis=1)
            logits = np.concatenate([start, logits], axis=0)
            path, _ = tf.contrib.crf.viterbi_decode(logits, transition_matrix)

            paths.append(path[1:])

        return paths

    def evaluate(self, sess, data):
        """
        :param sess: session  to run the model
        :param data: list of data
        :return: evaluate result
        """
        results = []
        trans = self.trans.eval(session=sess)

        dev_num = len(data.dev_Ids)
        total_batch = dev_num // self.batch_size

        for batch_id in range(total_batch):
            start = batch_id * self.batch_size
            end = (batch_id + 1) * self.batch_size

            if end > dev_num:
                end = dev_num

            instance = data.dev_Ids[start:end]
            if not instance:
                continue

            lengths, char_ids, lexicon_word_ids, word_length_tensor, target, _ = self.batch_with_label(instance)

            # run模型
            feed_dict = {
                self.placeholders["char_ids"]: char_ids,
                self.placeholders["lexicon_word_ids"]: lexicon_word_ids,
                self.placeholders["word_length_tensor"]: word_length_tensor,
            }

            logits = sess.run(self.logits, feed_dict=feed_dict)
            paths = self.decode(logits, lengths, trans)

            # confusion = confusion_matrix(list(chain.from_iterable(target)),
            #                              list(chain.from_iterable(paths)))

            tags = [data.label_alphabet.get_instance(idx) for idx in paths[0]]
            print("tags: ", tags)

        return results

    def evaluate_line(self, sess, sentence, data):
        '''
        因LatticeLSTM内部参数受batch_size限制，数据会转为批处理
        :param sess: 会话
        :param sentence: 带处理文本
        :param data: 含词库等处理的数据集
        :return: 返回标注结果
        '''
        _, Ids = data.generate_sentence_instance_with_gaz(sentence)
        lengths, char_ids, lexicon_word_ids, word_length_tensor = self.batch_with_label(Ids, False)

        lengths = lengths * self.batch_size
        char_ids = char_ids * self.batch_size
        lexicon_word_ids = lexicon_word_ids * self.batch_size
        word_length_tensor = word_length_tensor * self.batch_size

        # run模型
        feed_dict = {
            self.placeholders["char_ids"]: char_ids,
            self.placeholders["lexicon_word_ids"]: lexicon_word_ids,
            self.placeholders["word_length_tensor"]: word_length_tensor,
        }

        logits = sess.run(self.logits, feed_dict=feed_dict)
        paths = self.decode(logits, lengths, self.trans.eval(session=sess))
        tags = [data.label_alphabet.get_instance(idx) for idx in paths[0]]
        print("tags: ", tags)

        return tags


if __name__ == "__main__":
    model = LatticeNet()
    model.data = Data()
    model.load_data_and_embedding(model.data)
    model.create_model()
    model.train_model(model.data)

    model.save_tf_model()
