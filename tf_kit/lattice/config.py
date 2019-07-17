# _*_ coding: utf-8 _*_

BATCH_SIZE = 64
EMB_SIZE = 50
MAX_CHAR_LEN = 100
MAX_LEXICON_WORDS_NUM = 5
NUM_UNITS = 128
NUM_TAGS = 18
LEARNING_RATE = 0.005
CLIP = 5
OPTIMIZER = 'adam'
GAZ_FILE = 'data/ctb.50d.vec'
CHAR_EMB = 'data/gigaword_chn.all.a2b.uni.ite50.vec'
TRAIN_FILE = 'data/demo.train.char'
DEV_FILE = 'data/demo.dev.char'
TEST_FILE = 'data/demo.test.char'
MODEL_SAVE_PATH = 'model/ckpt'
